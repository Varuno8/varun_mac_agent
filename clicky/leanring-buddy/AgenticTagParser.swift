//
//  AgenticTagParser.swift
//  leanring-buddy
//
//  Hand-rolled state-machine parser for the agentic action grammar.
//  Supports: [CLICK] [DBLCLICK] [RCLICK] [AXCLICK] [TYPE] [HOTKEY] [SCROLL]
//            [APPLESCRIPT] [WAIT] [SCREENSHOT] [CONFIRM] [POINT] [TASK_DONE]
//            [PLAN] [SUBTASK_DONE]
//

import Foundation

// MARK: - Action Tag

enum ActionTag {
    case click(x: Int, y: Int, screen: Int?)
    case doubleClick(x: Int, y: Int, screen: Int?)
    case rightClick(x: Int, y: Int, screen: Int?)
    /// Click an accessibility element by its label/title/description.
    /// Bypasses pixel coordinates — the executor walks the frontmost app's
    /// AX tree to find the element and presses its actual location.
    case axClick(label: String)
    case type(text: String)
    case hotkey(keys: String)
    case scroll(direction: String, amount: Int, x: Int, y: Int, screen: Int?)
    case appleScript(source: String)
    case wait(milliseconds: Int)
    case screenshot
    case confirm(message: String)
    case point(x: Int, y: Int, label: String?, screen: Int?)
    case taskDone
    /// Registers a multi-step plan. Micky tracks progress and injects the
    /// current plan state into each subsequent turn so the model stays oriented.
    case plan(steps: [String])
    /// Marks a named plan step as complete so the plan block shows [x] for it.
    case subtaskDone(stepName: String)
}

// MARK: - Parse Result

struct ParsedAgenticResponse {
    /// All action tags found in the response, in order.
    let actions: [ActionTag]
    /// Response text with every recognized tag stripped — safe to pass to TTS.
    let spokenText: String
    /// Convenience: the first [POINT] tag, if any.
    var pointTag: ActionTag? {
        actions.first { if case .point = $0 { return true }; return false }
    }
}

// MARK: - Parser

enum AgenticTagParser {

    /// Parses the full response text and returns all action tags plus the
    /// spoken text with every tag stripped.
    static func parse(_ input: String) -> ParsedAgenticResponse {
        var actions: [ActionTag] = []
        var spokenParts: [String] = []
        var remaining = input[...]

        while !remaining.isEmpty {
            // Find the next opening bracket
            guard let bracketStart = remaining.firstIndex(of: "[") else {
                // No more tags — rest is spoken text
                spokenParts.append(String(remaining))
                break
            }

            // Everything before the bracket is spoken text
            let textBefore = String(remaining[remaining.startIndex..<bracketStart])
            if !textBefore.isEmpty {
                spokenParts.append(textBefore)
            }

            // Try to parse a complete tag starting at bracketStart
            let afterBracket = remaining.index(after: bracketStart)
            guard let bracketEnd = findMatchingClose(in: remaining, from: afterBracket) else {
                // No closing bracket — treat rest as spoken text
                spokenParts.append(String(remaining[bracketStart...]))
                break
            }

            let tagContent = String(remaining[afterBracket..<bracketEnd])
            let tagEnd = remaining.index(after: bracketEnd)

            if let action = parseTagContent(tagContent) {
                actions.append(action)
                // Don't add to spokenParts — tag is stripped from TTS text
            } else {
                // Unknown tag — preserve it as spoken text
                spokenParts.append("[\(tagContent)]")
            }

            remaining = remaining[tagEnd...]
        }

        let spokenText = spokenParts.joined()
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return ParsedAgenticResponse(actions: actions, spokenText: spokenText)
    }

    // MARK: - Private helpers

    /// Finds the next unescaped `]` starting at `from`, honoring `\]` escape.
    private static func findMatchingClose(in str: Substring, from start: String.Index) -> String.Index? {
        var idx = start
        while idx < str.endIndex {
            let ch = str[idx]
            if ch == "\\" {
                // Skip escaped character
                let next = str.index(after: idx)
                if next < str.endIndex { idx = str.index(after: next) } else { break }
            } else if ch == "]" {
                return idx
            } else {
                idx = str.index(after: idx)
            }
        }
        return nil
    }

    /// Parses the contents between `[` and `]`. Returns nil for unknown tags.
    private static func parseTagContent(_ content: String) -> ActionTag? {
        // Split on `:` but only for the tag name — payload may contain colons
        let colonIdx = content.firstIndex(of: ":") ?? content.endIndex
        let tagName = String(content[content.startIndex..<colonIdx]).uppercased()
        let payload = colonIdx < content.endIndex
            ? String(content[content.index(after: colonIdx)...])
            : ""

        switch tagName {
        case "CLICK":
            return parseCoordTag(payload: payload, make: { x, y, scr in .click(x: x, y: y, screen: scr) })
        case "DBLCLICK":
            return parseCoordTag(payload: payload, make: { x, y, scr in .doubleClick(x: x, y: y, screen: scr) })
        case "RCLICK":
            return parseCoordTag(payload: payload, make: { x, y, scr in .rightClick(x: x, y: y, screen: scr) })
        case "AXCLICK":
            let unescaped = payload.replacingOccurrences(of: "\\]", with: "]")
                .trimmingCharacters(in: .whitespaces)
            return unescaped.isEmpty ? nil : .axClick(label: unescaped)
        case "TYPE":
            let unescaped = payload.replacingOccurrences(of: "\\]", with: "]")
            return .type(text: unescaped)
        case "HOTKEY":
            return payload.isEmpty ? nil : .hotkey(keys: payload.lowercased())
        case "SCROLL":
            return parseScrollTag(payload: payload)
        case "APPLESCRIPT":
            let unescaped = payload.replacingOccurrences(of: "\\]", with: "]")
            return unescaped.isEmpty ? nil : .appleScript(source: unescaped)
        case "WAIT":
            guard let ms = Int(payload.trimmingCharacters(in: .whitespaces)) else { return nil }
            return .wait(milliseconds: ms)
        case "SCREENSHOT":
            return .screenshot
        case "CONFIRM":
            return .confirm(message: payload)
        case "POINT":
            return parsePointTag(payload: payload)
        case "TASK_DONE":
            return .taskDone
        case "PLAN":
            // Payload: "step one|step two|step three"
            let steps = payload
                .split(separator: "|", omittingEmptySubsequences: true)
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            return steps.isEmpty ? nil : .plan(steps: steps)
        case "SUBTASK_DONE":
            let stepName = payload.trimmingCharacters(in: .whitespaces)
            return stepName.isEmpty ? nil : .subtaskDone(stepName: stepName)
        default:
            return nil
        }
    }

    /// Parses `x,y` or `x,y:screenN` into a coordinate-based action tag.
    private static func parseCoordTag(payload: String, make: (Int, Int, Int?) -> ActionTag) -> ActionTag? {
        let parts = payload.split(separator: ":", maxSplits: 2, omittingEmptySubsequences: false)
        guard parts.count >= 1 else { return nil }
        let coords = parts[0].split(separator: ",")
        guard coords.count == 2,
              let x = Int(coords[0].trimmingCharacters(in: .whitespaces)),
              let y = Int(coords[1].trimmingCharacters(in: .whitespaces)) else { return nil }
        var screenNumber: Int? = nil
        if parts.count >= 2, let scr = Int(parts[1].trimmingCharacters(in: .whitespaces)) {
            screenNumber = scr
        }
        return make(x, y, screenNumber)
    }

    /// Parses `direction:amount:x,y` or `direction:amount:x,y:screenN`.
    private static func parseScrollTag(payload: String) -> ActionTag? {
        let parts = payload.split(separator: ":", maxSplits: 3, omittingEmptySubsequences: false)
        guard parts.count >= 3 else { return nil }
        let direction = String(parts[0]).lowercased()
        guard let amount = Int(parts[1].trimmingCharacters(in: .whitespaces)) else { return nil }
        let coords = parts[2].split(separator: ",")
        guard coords.count == 2,
              let x = Int(coords[0].trimmingCharacters(in: .whitespaces)),
              let y = Int(coords[1].trimmingCharacters(in: .whitespaces)) else { return nil }
        var screenNumber: Int? = nil
        if parts.count >= 4, let scr = Int(parts[3].trimmingCharacters(in: .whitespaces)) {
            screenNumber = scr
        }
        return .scroll(direction: direction, amount: amount, x: x, y: y, screen: screenNumber)
    }

    /// Parses `none` | `x,y` | `x,y:label` | `x,y:label:screenN`.
    private static func parsePointTag(payload: String) -> ActionTag? {
        if payload.lowercased() == "none" {
            return .point(x: 0, y: 0, label: nil, screen: nil)
        }
        let parts = payload.split(separator: ":", maxSplits: 2, omittingEmptySubsequences: false)
        guard parts.count >= 1 else { return nil }
        let coords = parts[0].split(separator: ",")
        guard coords.count == 2,
              let x = Int(coords[0].trimmingCharacters(in: .whitespaces)),
              let y = Int(coords[1].trimmingCharacters(in: .whitespaces)) else { return nil }
        var label: String? = nil
        var screenNumber: Int? = nil
        if parts.count >= 2 {
            label = String(parts[1]).trimmingCharacters(in: .whitespaces)
        }
        if parts.count >= 3, let scr = Int(parts[2].trimmingCharacters(in: .whitespaces)) {
            screenNumber = scr
        }
        return .point(x: x, y: y, label: label, screen: screenNumber)
    }
}
