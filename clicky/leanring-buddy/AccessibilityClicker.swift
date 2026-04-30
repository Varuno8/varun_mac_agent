//
//  AccessibilityClicker.swift
//  leanring-buddy
//
//  Walks the frontmost application's accessibility (AX) tree to find a UI
//  element matching a label and clicks its actual location — never depending
//  on pixel coordinates from a vision model.
//
//  Vision models (Gemini, GPT-4V, Claude) cannot place pixel coordinates
//  with sub-row precision in dense lists like WhatsApp's chat sidebar; they
//  routinely click 30-80 pixels off, hitting the row above or below the
//  intended one. AXCLICK eliminates that whole class of error by querying
//  macOS's accessibility services for the element by its accessible name
//  and pressing it directly via AXPress (or, as a fallback, via a CGEvent
//  click at the element's true center).
//

import AppKit
import ApplicationServices
import Foundation

@MainActor
enum AccessibilityClicker {

    /// Result of an AXCLICK attempt — either we hit something, or we describe
    /// what was visible so the caller can re-prompt the model with context.
    enum Result {
        case clicked(matchedLabel: String, role: String, viaAXPress: Bool)
        case noMatch(visibleLabelsSample: [String], frontApp: String)
        case axNotAvailable(reason: String)
    }

    /// Searches the frontmost app's AX tree for an element whose
    /// title/label/description/value contains `searchText` (case-insensitive),
    /// prefers actionable roles (button, cell, row, menu item), and clicks it.
    ///
    /// - Parameter searchText: substring to match against AX attributes.
    ///                         Case-insensitive. Whitespace-trimmed.
    /// - Returns: `.clicked` on success; `.noMatch` with a sample of visible
    ///            labels when nothing matched; `.axNotAvailable` when the
    ///            AX query couldn't be performed at all.
    static func click(searchText rawSearchText: String) -> Result {
        let searchText = rawSearchText
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard !searchText.isEmpty else {
            return .axNotAvailable(reason: "empty search text")
        }

        guard let frontApp = NSWorkspace.shared.frontmostApplication else {
            return .axNotAvailable(reason: "no frontmost application")
        }
        let frontAppName = frontApp.localizedName ?? "frontApp"

        let appElement = AXUIElementCreateApplication(frontApp.processIdentifier)

        // Walk the tree. We collect both matches and a sample of every visible
        // label so a no-match outcome can tell the model what *was* visible.
        var matches: [AXElementMatch] = []
        var visibleLabelsSample: [String] = []

        walkAXTree(
            root: appElement,
            maxDepth: 25,
            maxNodesVisited: 1500,
            visit: { axElement, attributes in
                // Track every non-empty label so a no-match can show what was visible
                let allLabels = [attributes.title, attributes.description, attributes.value, attributes.identifier]
                    .compactMap { $0 }
                    .filter { !$0.isEmpty && $0.count <= 80 }
                visibleLabelsSample.append(contentsOf: allLabels)

                // Match: any label contains the search text (case-insensitive)
                guard let matchedLabel = allLabels.first(where: { $0.lowercased().contains(searchText) }) else {
                    return
                }
                matches.append(AXElementMatch(
                    element: axElement,
                    matchedLabel: matchedLabel,
                    role: attributes.role ?? "",
                    isExactMatch: matchedLabel.lowercased() == searchText
                ))
            }
        )

        guard let bestMatch = pickBestMatch(from: matches) else {
            // De-duplicate and trim the visible labels for the no-match report
            let dedupedSample = Array(Set(visibleLabelsSample)).sorted().prefix(40).map { $0 }
            return .noMatch(visibleLabelsSample: dedupedSample, frontApp: frontAppName)
        }

        // Try AXPress first — it works for any element with a Press action and
        // doesn't need a frame. Cells/rows often DON'T have Press; they need
        // an explicit click at their center.
        let pressResult = AXUIElementPerformAction(bestMatch.element, kAXPressAction as CFString)
        if pressResult == .success {
            return .clicked(matchedLabel: bestMatch.matchedLabel, role: bestMatch.role, viaAXPress: true)
        }

        // Fall back to CGEvent click at the element's actual center.
        guard let frame = frame(of: bestMatch.element), frame.width > 0, frame.height > 0 else {
            return .axNotAvailable(
                reason: "found '\(bestMatch.matchedLabel)' but couldn't read its frame (AXPress also failed)"
            )
        }
        let centerInScreenCoords = CGPoint(x: frame.midX, y: frame.midY)
        postCGEventClick(at: centerInScreenCoords)
        return .clicked(matchedLabel: bestMatch.matchedLabel, role: bestMatch.role, viaAXPress: false)
    }

    // MARK: - Private types

    private struct AXElementMatch {
        let element: AXUIElement
        let matchedLabel: String
        let role: String
        let isExactMatch: Bool
    }

    private struct AXElementAttributes {
        let role: String?
        let title: String?
        let description: String?
        let value: String?
        let identifier: String?
    }

    // MARK: - Tree walking

    /// Iterative BFS over the AX tree. Caps depth and total nodes visited so
    /// pathological trees (Electron apps with thousands of divs) don't hang.
    private static func walkAXTree(
        root: AXUIElement,
        maxDepth: Int,
        maxNodesVisited: Int,
        visit: (AXUIElement, AXElementAttributes) -> Void
    ) {
        var queue: [(AXUIElement, Int)] = [(root, 0)]
        var nodesVisited = 0

        while !queue.isEmpty && nodesVisited < maxNodesVisited {
            let (element, depth) = queue.removeFirst()
            nodesVisited += 1
            if depth > maxDepth { continue }

            let attributes = AXElementAttributes(
                role:        copyStringAttribute(element, kAXRoleAttribute),
                title:       copyStringAttribute(element, kAXTitleAttribute),
                description: copyStringAttribute(element, kAXDescriptionAttribute),
                value:       copyStringAttribute(element, kAXValueAttribute),
                identifier:  copyStringAttribute(element, kAXIdentifierAttribute)
            )
            visit(element, attributes)

            if let children = copyChildren(element) {
                for child in children {
                    queue.append((child, depth + 1))
                }
            }
        }
    }

    /// Score-based pick: actionable roles (button, cell, row, menu item)
    /// outrank static text; exact label match outranks substring.
    private static func pickBestMatch(from matches: [AXElementMatch]) -> AXElementMatch? {
        guard !matches.isEmpty else { return nil }

        // Role string literals — most have kAX*Role constants but not all
        // (e.g., kAXLinkRole doesn't exist; the role string is just "AXLink").
        let rolePriority: [String: Int] = [
            kAXButtonRole as String:      100,
            kAXMenuItemRole as String:     90,
            "AXLink":                      85,
            kAXCellRole as String:         80,
            kAXRowRole as String:          75,
            kAXMenuButtonRole as String:   70,
            kAXCheckBoxRole as String:     65,
            kAXRadioButtonRole as String:  65,
            kAXTextFieldRole as String:    60,
            kAXTabGroupRole as String:     50,
            kAXStaticTextRole as String:   30,
            kAXImageRole as String:        20,
        ]

        return matches.max(by: { lhs, rhs in
            let lhsScore = (rolePriority[lhs.role] ?? 40) + (lhs.isExactMatch ? 50 : 0)
            let rhsScore = (rolePriority[rhs.role] ?? 40) + (rhs.isExactMatch ? 50 : 0)
            return lhsScore < rhsScore
        })
    }

    // MARK: - AX accessor helpers

    private static func copyStringAttribute(_ element: AXUIElement, _ attribute: String) -> String? {
        var ref: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(element, attribute as CFString, &ref)
        guard result == .success, let value = ref else { return nil }
        if let str = value as? String { return str }
        // AXValue numeric types — not what we want for label matching
        return nil
    }

    private static func copyChildren(_ element: AXUIElement) -> [AXUIElement]? {
        var ref: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(element, kAXChildrenAttribute as CFString, &ref)
        guard result == .success, let value = ref else { return nil }
        return value as? [AXUIElement]
    }

    /// AX positions/sizes come back as AXValue refs. We unbox them into a
    /// CGRect in screen coordinates (top-left origin, points).
    private static func frame(of element: AXUIElement) -> CGRect? {
        var positionRef: CFTypeRef?
        var sizeRef: CFTypeRef?
        let positionResult = AXUIElementCopyAttributeValue(element, kAXPositionAttribute as CFString, &positionRef)
        let sizeResult     = AXUIElementCopyAttributeValue(element, kAXSizeAttribute as CFString, &sizeRef)
        guard positionResult == .success, sizeResult == .success,
              let positionRef, let sizeRef else { return nil }

        // Force-cast is safe here: kAXPositionAttribute always returns AXValue
        // wrapping a CGPoint when the element exposes a position at all.
        let positionValue = positionRef as! AXValue
        let sizeValue     = sizeRef as! AXValue

        var origin = CGPoint.zero
        var size   = CGSize.zero
        guard AXValueGetValue(positionValue, .cgPoint, &origin),
              AXValueGetValue(sizeValue, .cgSize, &size) else {
            return nil
        }
        return CGRect(origin: origin, size: size)
    }

    // MARK: - CGEvent fallback

    private static func postCGEventClick(at point: CGPoint) {
        let source = CGEventSource(stateID: .hidSystemState)
        let down = CGEvent(mouseEventSource: source, mouseType: .leftMouseDown, mouseCursorPosition: point, mouseButton: .left)
        let up   = CGEvent(mouseEventSource: source, mouseType: .leftMouseUp,   mouseCursorPosition: point, mouseButton: .left)
        down?.post(tap: .cghidEventTap)
        up?.post(tap: .cghidEventTap)
    }
}
