//
//  AgenticActionExecutor.swift
//  leanring-buddy
//
//  Executes parsed agentic action tags via CGEvent (mouse/keyboard),
//  NSAppleScript, and NSWorkspace. Runs on MainActor; AppleScript
//  dispatches to a background queue with a 10s timeout.
//

import AppKit
import Carbon
import Foundation

// MARK: - Execution Outcome

enum AgenticExecutionOutcome {
    case completed
    /// [SCREENSHOT] tag was reached. executedSummary lists what ran before the screenshot tag.
    case needsScreenshotRefresh(executedSummary: String)
    case cancelled
    case blockedAwaitingConfirmation(message: String, pendingActions: [ActionTag])
}

// MARK: - Installed Application Catalog

/// Discovers macOS apps installed on this machine and resolves a fuzzy app-name
/// query (whatever Gemini wrote in `open -a "..."`) to the closest real bundle
/// name. No hardcoded alias map — every match comes from what's actually on disk.
/// Built once per process, cached.
///
/// Resolution order: exact case-insensitive match → substring match (shortest
/// containing app wins) → token-AND match (every word in the query appears
/// somewhere in the app name) → Levenshtein within a tight tolerance. Returns
/// nil if nothing reasonable is found, in which case the caller should leave
/// the original command alone and let LaunchServices surface the error.
enum InstalledApplicationCatalog {

    static let allInstalledAppNames: [String] = scanInstalledApps()

    private static func scanInstalledApps() -> [String] {
        let realUserHome = NSHomeDirectoryForUser(NSUserName()) ?? NSHomeDirectory()
        let applicationFolders = [
            "/Applications",
            "/System/Applications",
            "/System/Applications/Utilities",
            "\(realUserHome)/Applications",
        ]
        var discoveredAppNames: Set<String> = []
        for folder in applicationFolders {
            guard let entries = try? FileManager.default.contentsOfDirectory(atPath: folder) else { continue }
            for entry in entries where entry.hasSuffix(".app") {
                discoveredAppNames.insert(String(entry.dropLast(".app".count)))
            }
        }
        return discoveredAppNames.sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }
    }

    /// Best-effort resolve `query` to an installed app name, or nil if nothing is close enough.
    static func resolve(_ query: String) -> String? {
        let trimmedQuery = query.trimmingCharacters(in: .whitespaces)
        guard !trimmedQuery.isEmpty else { return nil }
        let lowercaseQuery = trimmedQuery.lowercased()
        let installedAppNames = allInstalledAppNames

        // 1. Exact case-insensitive match — preferred when present.
        if let exactMatch = installedAppNames.first(where: { $0.lowercased() == lowercaseQuery }) {
            return exactMatch
        }

        // 2. Substring match — pick the shortest installed app name that contains
        //    the query. Shortest wins so "Code" prefers a hypothetical "Code.app"
        //    over "Visual Studio Code - Insiders.app".
        let substringMatches = installedAppNames.filter { $0.lowercased().contains(lowercaseQuery) }
        if let shortestSubstringMatch = substringMatches.min(by: { $0.count < $1.count }) {
            return shortestSubstringMatch
        }

        // 3. Token-AND match — split the query into words and require every word
        //    to appear (as a substring) in the candidate app name. Catches
        //    "vs code" → "Visual Studio Code - Insiders" because both "vs" and
        //    "code" are substrings of that name.
        let queryTokens = lowercaseQuery
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map(String.init)
            .filter { $0.count >= 2 }
        if queryTokens.count >= 2 {
            let tokenMatches = installedAppNames.filter { appName in
                let lowercaseAppName = appName.lowercased()
                return queryTokens.allSatisfy { lowercaseAppName.contains($0) }
            }
            if let shortestTokenMatch = tokenMatches.min(by: { $0.count < $1.count }) {
                return shortestTokenMatch
            }
        }

        // 4. Levenshtein fallback — only accept matches within ~25% edit distance
        //    of the query length (or 2, whichever is larger). Stops "antigravity"
        //    from being mapped to "Activity Monitor" just because they share a few letters.
        let fuzzyTolerance = max(2, lowercaseQuery.count / 4)
        let fuzzyMatch = installedAppNames
            .map { ($0, levenshteinDistance($0.lowercased(), lowercaseQuery)) }
            .filter { $0.1 <= fuzzyTolerance }
            .min(by: { $0.1 < $1.1 })
        return fuzzyMatch?.0
    }

    /// Standard iterative-DP Levenshtein distance. Two-row implementation to keep
    /// allocation small even when scanning ~hundreds of installed apps per query.
    private static func levenshteinDistance(_ stringA: String, _ stringB: String) -> Int {
        let charactersA = Array(stringA)
        let charactersB = Array(stringB)
        if charactersA.isEmpty { return charactersB.count }
        if charactersB.isEmpty { return charactersA.count }
        var previousRow = Array(0...charactersB.count)
        var currentRow = Array(repeating: 0, count: charactersB.count + 1)
        for i in 1...charactersA.count {
            currentRow[0] = i
            for j in 1...charactersB.count {
                let substitutionCost = charactersA[i - 1] == charactersB[j - 1] ? 0 : 1
                currentRow[j] = min(
                    currentRow[j - 1] + 1,
                    previousRow[j] + 1,
                    previousRow[j - 1] + substitutionCost
                )
            }
            (previousRow, currentRow) = (currentRow, previousRow)
        }
        return previousRow[charactersB.count]
    }
}

// MARK: - Executor

@MainActor
final class AgenticActionExecutor {

    static let maxActionsPerTurn = 20
    static let interActionDelayMs: UInt64 = 80
    static let typeCharacterDelayMs: UInt64 = 12

    private var isCancelled = false
    private var escMonitor: Any?

    // Actual pixel dimensions of the most recent screenshot sent to Gemini.
    // Must be set before execute() so coordinate mapping is exact.
    var lastScreenshotWidth: Int = 1280
    var lastScreenshotHeight: Int = 800

    // Raw image data of the most recent screenshot. Set by CompanionManager
    // before execute() so OCR fallback for [AXCLICK] can find text in the
    // current screen state without an extra capture round-trip.
    var lastScreenshotImageData: Data?

    // Tracks whether the most recent CLICK/DBLCLICK/RCLICK in the current
    // execution batch has been visually verified by a subsequent [SCREENSHOT].
    // Set true on any click. Reset to false when a [SCREENSHOT] is reached
    // (the executor returns at that point so the next chat round starts with
    // a fresh screenshot — i.e. verification). Reset at the start of each
    // execute() call so per-turn state never leaks across turns.
    //
    // While true, [TYPE] and [HOTKEY:return]/[HOTKEY:enter] are refused —
    // the executor returns .needsScreenshotRefresh with a "BLOCKED:" prefix
    // in the summary so CompanionManager can re-prompt Gemini with a system
    // note explaining why. This is the hard backstop for wrong-recipient
    // bugs that prompt rules alone don't reliably prevent.
    private var hasUnverifiedClick = false

    // Patterns that indicate unambiguously dangerous shell operations.
    // Kept tight to avoid false positives — "delete" and "trash" are omitted
    // because they appear in legitimate AppleScript (Finder delete, Trash folder).
    // Patterns like "| bash" and "eval $(" catch execution-hijacking injection.
    private static let destructiveKeywords = [
        "rm -rf", "sudo rm", "sudo dd", "dd if=",
        "mkfs", "diskutil eraseDisk", "diskutil zeroDisk",
        "shutdown", "reboot", "poweroff",
        "| bash", "| sh", "eval $(", "`curl", "`wget",
    ]

    func execute(
        actions: [ActionTag],
        onActionStart: @escaping (String) -> Void,
        onActionFinish: @escaping () -> Void
    ) async -> AgenticExecutionOutcome {

        isCancelled = false
        hasUnverifiedClick = false
        installEscMonitor()
        defer { removeEscMonitor() }

        let cappedActions = Array(actions.prefix(Self.maxActionsPerTurn))
        var executionLog: [String] = []

        for (index, action) in cappedActions.enumerated() {
            guard !isCancelled else { return .cancelled }

            // Check destructive heuristic before executing
            if let destructiveWarning = destructiveWarning(for: action) {
                let remaining = Array(cappedActions.dropFirst(index))
                return .blockedAwaitingConfirmation(message: destructiveWarning, pendingActions: remaining)
            }

            // Hard backstop for wrong-target bugs: refuse [TYPE] and
            // [HOTKEY:return]/[HOTKEY:enter] when the most recent click hasn't
            // been verified by an intervening [SCREENSHOT]. This is the
            // executor-level enforcement of the prompt rule the model
            // sometimes ignores (causing "wrong recipient" message sends).
            if let blockedReason = unverifiedClickBlockReason(for: action) {
                let summaryWithBlock = (executionLog + [blockedReason])
                    .joined(separator: "; ")
                print("🛑 \(blockedReason) — auto-screenshot for re-plan")
                onActionFinish()
                return .needsScreenshotRefresh(executedSummary: summaryWithBlock)
            }

            let description = actionDescription(for: action)
            onActionStart(description)

            switch action {
            case .click(let x, let y, let screen):
                postMouseClick(x: x, y: y, screen: screen, clickCount: 1, button: .left)
                executionLog.append("clicked at (\(x),\(y))")
                hasUnverifiedClick = true

            case .doubleClick(let x, let y, let screen):
                postMouseClick(x: x, y: y, screen: screen, clickCount: 2, button: .left)
                executionLog.append("double-clicked at (\(x),\(y))")
                hasUnverifiedClick = true

            case .rightClick(let x, let y, let screen):
                postMouseClick(x: x, y: y, screen: screen, clickCount: 1, button: .right)
                executionLog.append("right-clicked at (\(x),\(y))")
                hasUnverifiedClick = true

            case .axClick(let label):
                let outcome = await executeAXClickWithOCRFallback(label: label, executionLog: &executionLog)
                if let blockingOutcome = outcome {
                    onActionFinish()
                    return blockingOutcome
                }

            case .type(let text):
                // Bring the frontmost app to focus before typing so keystrokes land in the right window
                NSWorkspace.shared.frontmostApplication?.activate(options: .activateIgnoringOtherApps)
                try? await Task.sleep(nanoseconds: 150_000_000)
                await typeText(text)
                executionLog.append("typed \"\(text.prefix(40))\"")

            case .hotkey(let keys):
                postHotkey(keys: keys)
                executionLog.append("pressed \(keys)")

            case .scroll(let direction, let amount, let x, let y, let screen):
                postScroll(direction: direction, amount: amount, x: x, y: y, screen: screen)
                executionLog.append("scrolled \(direction) \(amount)x")

            case .appleScript(let source):
                // Rewrite chain, in order:
                //   1. activate-style `tell application "X" to activate` → `do shell script "open -a X"`
                //   2. fuzzy-resolve every `open -a NAME` to a real installed app (no hardcoded aliases)
                //   3. expand `~` to the real user home so paths don't resolve to the sandbox container
                let rewrittenSource = expandTildesInShellScript(
                    resolveAppNamesInShellScript(
                        rewriteActivateToOpenShell(source)
                    )
                )
                let result = await runAppleScript(source: rewrittenSource)
                switch result {
                case .success(let output):
                    let shortSource = String(rewrittenSource.prefix(60))
                    let note = output.isEmpty ? "" : " → \"\(output.prefix(40))\""
                    executionLog.append("ran AppleScript: \(shortSource)\(note)")
                case .error(let msg):
                    print("⚠️ AppleScript error: \(msg)")
                    executionLog.append("AppleScript FAILED (\(msg)): \(String(rewrittenSource.prefix(60)))")
                }

            case .wait(let ms):
                try? await Task.sleep(nanoseconds: UInt64(ms) * 1_000_000)
                executionLog.append("waited \(ms)ms")

            case .screenshot:
                hasUnverifiedClick = false
                onActionFinish()
                let summary = executionLog.isEmpty ? "no actions run yet" : executionLog.joined(separator: "; ")
                return .needsScreenshotRefresh(executedSummary: summary)

            case .confirm(let message):
                let remaining = Array(cappedActions.dropFirst(index + 1))
                return .blockedAwaitingConfirmation(message: message, pendingActions: remaining)

            case .point:
                // Handled by CompanionManager — not executed here
                break

            case .taskDone:
                // Handled by CompanionManager — signals task completion, not executed here
                break

            case .plan(let steps):
                // Handled by CompanionManager — plan state is tracked and injected per-turn
                executionLog.append("plan registered: \(steps.count) steps")

            case .subtaskDone(let stepName):
                // Handled by CompanionManager — updates the plan's completion state
                executionLog.append("subtask done: \(stepName)")
            }

            onActionFinish()

            // Small inter-action delay so the OS can process each event
            if index < cappedActions.count - 1 {
                try? await Task.sleep(nanoseconds: Self.interActionDelayMs * 1_000_000)
            }
        }

        return .completed
    }

    // MARK: - Esc kill-switch

    private func installEscMonitor() {
        escMonitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { [weak self] event in
            if event.keyCode == 53 { // Esc
                self?.isCancelled = true
            }
        }
    }

    private func removeEscMonitor() {
        if let monitor = escMonitor {
            NSEvent.removeMonitor(monitor)
            escMonitor = nil
        }
    }

    // MARK: - Mouse

    private func postMouseClick(x: Int, y: Int, screen: Int?, clickCount: Int, button: CGMouseButton) {
        let cgPoint = AgenticCoordinateMapper.cgEventGlobal(
            screenshotX: x, screenshotY: y,
            actualScreenshotWidth: lastScreenshotWidth,
            actualScreenshotHeight: lastScreenshotHeight,
            screenIndex: screen
        )
        print("🖱️ Click: screenshot(\(x),\(y)) → cgEvent(\(Int(cgPoint.x)),\(Int(cgPoint.y)))")
        let downType: CGEventType = button == .left ? .leftMouseDown : .rightMouseDown
        let upType: CGEventType = button == .left ? .leftMouseUp : .rightMouseUp

        let src = CGEventSource(stateID: .hidSystemState)
        let down = CGEvent(mouseEventSource: src, mouseType: downType, mouseCursorPosition: cgPoint, mouseButton: button)
        let up   = CGEvent(mouseEventSource: src, mouseType: upType,   mouseCursorPosition: cgPoint, mouseButton: button)

        if clickCount == 2 {
            down?.setIntegerValueField(.mouseEventClickState, value: 2)
            up?.setIntegerValueField(.mouseEventClickState, value: 2)
        }

        down?.post(tap: .cghidEventTap)
        up?.post(tap: .cghidEventTap)
    }

    // MARK: - Typing

    private func typeText(_ text: String) async {
        for scalar in text.unicodeScalars {
            guard !isCancelled else { return }
            let src = CGEventSource(stateID: .hidSystemState)
            let keyDown = CGEvent(keyboardEventSource: src, virtualKey: 0, keyDown: true)
            let keyUp   = CGEvent(keyboardEventSource: src, virtualKey: 0, keyDown: false)
            // CGEvent requires UTF-16 (UniChar = UInt16), not UTF-32 scalar values
            var chars = Array(String(scalar).utf16)
            keyDown?.keyboardSetUnicodeString(stringLength: chars.count, unicodeString: &chars)
            keyUp?.keyboardSetUnicodeString(stringLength: chars.count, unicodeString: &chars)
            keyDown?.post(tap: .cghidEventTap)
            keyUp?.post(tap: .cghidEventTap)
            try? await Task.sleep(nanoseconds: Self.typeCharacterDelayMs * 1_000_000)
        }
    }

    // MARK: - Hotkeys

    private func postHotkey(keys: String) {
        // Parse "command+shift+t" into modifier flags + key
        let parts = keys.split(separator: "+").map { String($0).lowercased() }
        var flags: CGEventFlags = []
        var keyChar: String = ""

        for part in parts {
            switch part {
            case "command", "cmd":   flags.insert(.maskCommand)
            case "shift":            flags.insert(.maskShift)
            case "option", "alt":    flags.insert(.maskAlternate)
            case "control", "ctrl":  flags.insert(.maskControl)
            default: keyChar = part
            }
        }

        guard let virtualKey = virtualKeyCode(for: keyChar) else {
            print("⚠️ Unknown hotkey character: \(keyChar)")
            return
        }

        let src = CGEventSource(stateID: .hidSystemState)
        let down = CGEvent(keyboardEventSource: src, virtualKey: virtualKey, keyDown: true)
        let up   = CGEvent(keyboardEventSource: src, virtualKey: virtualKey, keyDown: false)
        down?.flags = flags
        up?.flags = flags
        down?.post(tap: .cghidEventTap)
        up?.post(tap: .cghidEventTap)
    }

    // MARK: - Scroll

    private func postScroll(direction: String, amount: Int, x: Int, y: Int, screen: Int?) {
        let cgPoint = AgenticCoordinateMapper.cgEventGlobal(
            screenshotX: x, screenshotY: y,
            actualScreenshotWidth: lastScreenshotWidth,
            actualScreenshotHeight: lastScreenshotHeight,
            screenIndex: screen
        )
        let src = CGEventSource(stateID: .hidSystemState)

        // Move mouse to scroll target first
        let move = CGEvent(mouseEventSource: src, mouseType: .mouseMoved, mouseCursorPosition: cgPoint, mouseButton: .left)
        move?.post(tap: .cghidEventTap)

        let deltaY: Int32
        let deltaX: Int32
        switch direction {
        case "up":    deltaY = Int32(amount);  deltaX = 0
        case "down":  deltaY = -Int32(amount); deltaX = 0
        case "left":  deltaY = 0; deltaX = Int32(amount)
        case "right": deltaY = 0; deltaX = -Int32(amount)
        default:      deltaY = 0; deltaX = 0
        }

        let scroll = CGEvent(scrollWheelEvent2Source: src, units: .line, wheelCount: 2, wheel1: deltaY, wheel2: deltaX, wheel3: 0)
        scroll?.post(tap: .cghidEventTap)
    }

    // MARK: - AppleScript

    /// Finds every `open -a NAME` directive in the AppleScript source and rewrites
    /// NAME to the closest installed app via `InstalledApplicationCatalog.resolve`.
    /// If the model wrote a name that's already installed exactly, this is a no-op.
    /// If the model wrote a name with no reasonable match, the directive is left
    /// unchanged so LaunchServices returns a real error instead of a silent miss.
    private func resolveAppNamesInShellScript(_ source: String) -> String {
        // Collect every quoted or unquoted app name appearing after `open -a`.
        // We look up unique names once and then do plain string replacement, which
        // keeps the rewrite deterministic regardless of how many times the same
        // app appears in the script.
        var queriedAppNames: Set<String> = []

        let quotedAppArgumentPattern = #/open\s+-a\s+(?:'([^']+)'|"([^"]+)")/#
        for match in source.matches(of: quotedAppArgumentPattern) {
            let capturedName = match.1 ?? match.2
            if let capturedName, !capturedName.isEmpty {
                queriedAppNames.insert(String(capturedName))
            }
        }

        // Unquoted form: `open -a Word` where Word is a single token (no spaces).
        // We only match a conservative character class so we don't grab flags
        // like `-W` or path arguments that follow.
        let unquotedAppArgumentPattern = #/open\s+-a\s+([A-Za-z][A-Za-z0-9._-]*)/#
        for match in source.matches(of: unquotedAppArgumentPattern) {
            queriedAppNames.insert(String(match.1))
        }

        var rewrittenSource = source
        for queriedName in queriedAppNames {
            guard let resolvedName = InstalledApplicationCatalog.resolve(queriedName),
                  resolvedName != queriedName else {
                continue
            }
            // Replace all four common forms the model emits, normalising to single quotes.
            rewrittenSource = rewrittenSource
                .replacingOccurrences(of: "open -a '\(queriedName)'",   with: "open -a '\(resolvedName)'")
                .replacingOccurrences(of: "open -a \"\(queriedName)\"", with: "open -a '\(resolvedName)'")
                .replacingOccurrences(of: "open -a \(queriedName) ",    with: "open -a '\(resolvedName)' ")
                .replacingOccurrences(of: "open -a \(queriedName)\"",   with: "open -a '\(resolvedName)'\"")
            print("🔧 Resolved app name '\(queriedName)' → '\(resolvedName)'")
        }
        return rewrittenSource
    }

    /// Rewrites "tell application \"X\" to activate" → do shell script "open -a X"
    /// because `tell ... to activate` often fails for apps not registered as scriptable.
    /// Handles single-line and multi-line variants. Leaves all other AppleScript untouched.
    private func rewriteActivateToOpenShell(_ source: String) -> String {
        // Match: tell application "AppName" to activate
        // or:    tell application "AppName"\n    activate\nend tell
        let singleLinePattern = #/tell\s+application\s+"([^"]+)"\s+to\s+activate/#
        if let match = source.firstMatch(of: singleLinePattern) {
            let appName = String(match.1)
            let rewritten = #"do shell script "open -a '\#(appName)'""#
            print("🔧 Rewrote activate → open -a: \(appName)")
            return rewritten
        }

        // Match multi-line: tell application "AppName" \n activate \n end tell
        let multiLinePattern = #/tell\s+application\s+"([^"]+)"[\s\S]*?\bactivate\b[\s\S]*?end\s+tell/#
        if let match = source.firstMatch(of: multiLinePattern) {
            let appName = String(match.1)
            let rewritten = #"do shell script "open -a '\#(appName)'""#
            print("🔧 Rewrote multi-line activate → open -a: \(appName)")
            return rewritten
        }

        return source
    }

    /// Replaces `~/` with the real user home directory before the AppleScript runs.
    /// Without this, a sandboxed build resolves `~` to its container path
    /// (e.g. `~/Library/Containers/.../Data/Downloads/...`), causing `open -a` to fail.
    /// `NSHomeDirectoryForUser(NSUserName())` returns the real home even under sandbox.
    private func expandTildesInShellScript(_ source: String) -> String {
        let realUserHome = NSHomeDirectoryForUser(NSUserName()) ?? NSHomeDirectory()
        var result = source
        let tildeContexts: [(String, String)] = [
            (" ~/",  " \(realUserHome)/"),
            ("'~/",  "'\(realUserHome)/"),
            ("\"~/", "\"\(realUserHome)/"),
            ("=~/",  "=\(realUserHome)/"),
        ]
        for (matchPrefix, replacementPrefix) in tildeContexts {
            result = result.replacingOccurrences(of: matchPrefix, with: replacementPrefix)
        }
        if result != source {
            print("🔧 Expanded ~ to \(realUserHome) in AppleScript")
        }
        return result
    }

    enum AppleScriptResult {
        case success(String)
        case error(String)
    }

    private func runAppleScript(source: String) async -> AppleScriptResult {
        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                var error: NSDictionary?
                let script = NSAppleScript(source: source)
                let result = script?.executeAndReturnError(&error)

                if let error {
                    let msg = error[NSAppleScript.errorMessage] as? String ?? "unknown error"
                    continuation.resume(returning: .error(msg))
                } else {
                    continuation.resume(returning: .success(result?.stringValue ?? ""))
                }
            }
        }
    }

    // MARK: - AXCLICK with OCR fallback

    /// Tries Accessibility tree click first; if no AX match, falls back to
    /// Vision-framework OCR on the latest screenshot. Returns nil on success
    /// (caller continues with next action). Returns a `.needsScreenshotRefresh`
    /// outcome when both AX and OCR fail to find the target — caller should
    /// return that immediately so Gemini gets a re-plan opportunity.
    private func executeAXClickWithOCRFallback(
        label: String,
        executionLog: inout [String]
    ) async -> AgenticExecutionOutcome? {
        // Step 1: try Accessibility tree
        let axResult = AccessibilityClicker.click(searchText: label)
        switch axResult {
        case .clicked(let matchedLabel, let role, let viaAXPress):
            let mechanism = viaAXPress ? "AXPress" : "click@center"
            print("✅ AXCLICK: matched '\(matchedLabel)' (\(role)) via \(mechanism)")
            executionLog.append("axclicked '\(matchedLabel)' (\(role))")
            hasUnverifiedClick = true
            return nil

        case .noMatch(let visibleLabelsSample, let frontApp):
            // Fall through to OCR fallback below, but keep the AX diagnostic
            // so a final block message can show what AX *did* see.
            let axSampleSummary = visibleLabelsSample.prefix(15).joined(separator: " | ")
            return await fallbackToOCRClick(
                label: label,
                executionLog: &executionLog,
                axDiagnostic: "AX in \(frontApp) saw: \(axSampleSummary)"
            )

        case .axNotAvailable(let reason):
            return await fallbackToOCRClick(
                label: label,
                executionLog: &executionLog,
                axDiagnostic: "AX unavailable: \(reason)"
            )
        }
    }

    /// Vision-framework OCR fallback: searches the latest screenshot for
    /// `label` (case-insensitive substring) and clicks the center of the
    /// matching text's bounding box. Returns nil on success or a blocking
    /// outcome when OCR also can't find the text.
    private func fallbackToOCRClick(
        label: String,
        executionLog: inout [String],
        axDiagnostic: String
    ) async -> AgenticExecutionOutcome? {
        guard let screenshotImageData = lastScreenshotImageData else {
            let blockReason = "BLOCKED: AXCLICK couldn't find \"\(label)\" and OCR fallback " +
                "has no screenshot to search. \(axDiagnostic). " +
                "try [SCREENSHOT] then re-plan."
            print("🛑 \(blockReason)")
            executionLog.append(blockReason)
            return .needsScreenshotRefresh(executedSummary: executionLog.joined(separator: "; "))
        }

        let ocrResult = OCRClicker.click(
            searchText: label,
            screenshotData: screenshotImageData,
            screenshotWidth: lastScreenshotWidth,
            screenshotHeight: lastScreenshotHeight,
            screenIndex: nil
        )

        switch ocrResult {
        case .clicked(let matchedText, let displayPoint):
            print("✅ OCR fallback clicked '\(matchedText)' at (\(Int(displayPoint.x)),\(Int(displayPoint.y)))")
            executionLog.append("ocr-clicked '\(matchedText)' at (\(Int(displayPoint.x)),\(Int(displayPoint.y)))")
            hasUnverifiedClick = true
            return nil

        case .noMatch(let visibleTextSample):
            let ocrSampleSummary = visibleTextSample.prefix(20).joined(separator: " | ")
            let blockReason = "BLOCKED: neither AXCLICK nor OCR found \"\(label)\". " +
                "\(axDiagnostic). OCR saw text: \(ocrSampleSummary). " +
                "the element might not be visible — scroll, switch app, or use a more specific label."
            print("🛑 \(blockReason)")
            executionLog.append(blockReason)
            return .needsScreenshotRefresh(executedSummary: executionLog.joined(separator: "; "))

        case .failed(let reason):
            let blockReason = "BLOCKED: AXCLICK had no match and OCR failed (\(reason)). " +
                "\(axDiagnostic). try [SCREENSHOT] and re-plan."
            print("🛑 \(blockReason)")
            executionLog.append(blockReason)
            return .needsScreenshotRefresh(executedSummary: executionLog.joined(separator: "; "))
        }
    }

    // MARK: - Unverified-click backstop

    /// Returns a non-nil "BLOCKED:" reason string when the executor should
    /// refuse the given action because the prior click hasn't been verified
    /// by an intervening [SCREENSHOT]. Returns nil otherwise.
    ///
    /// The blocked actions are exactly the ones that *commit* a click into
    /// real user-visible side effects:
    ///   - [TYPE]                       — could fill the wrong field
    ///   - [HOTKEY:return] / [HOTKEY:enter] — could submit a form / send a
    ///                                       message to the wrong recipient
    /// Other hotkeys (cmd+a, escape, tab, etc.) are allowed because they
    /// don't commit; if the prior click was wrong they're either harmless or
    /// recoverable.
    private func unverifiedClickBlockReason(for action: ActionTag) -> String? {
        guard hasUnverifiedClick else { return nil }
        switch action {
        case .type(let text):
            let preview = String(text.prefix(40))
            return "BLOCKED: refused to TYPE \"\(preview)\" because the " +
                   "prior CLICK wasn't verified by a [SCREENSHOT] — auto-" +
                   "verifying now. Look at the new screenshot, confirm the " +
                   "click landed on the right target (correct chat header, " +
                   "right form field, etc.), and either re-click or proceed."
        case .hotkey(let keys):
            let lowercaseKeys = keys.lowercased()
            guard lowercaseKeys == "return" || lowercaseKeys == "enter" else { return nil }
            return "BLOCKED: refused to press \(keys) because the prior " +
                   "CLICK wasn't verified by a [SCREENSHOT] — auto-" +
                   "verifying now. Look at the new screenshot, confirm the " +
                   "click landed on the right target before submitting."
        default:
            return nil
        }
    }

    // MARK: - Destructive heuristic

    private func destructiveWarning(for action: ActionTag) -> String? {
        switch action {
        case .type(let text):
            let lower = text.lowercased()
            for keyword in Self.destructiveKeywords {
                if lower.contains(keyword) {
                    return "This will type \"\(text)\" which looks destructive. Are you sure?"
                }
            }
        case .appleScript(let source):
            let lower = source.lowercased()
            for keyword in Self.destructiveKeywords {
                if lower.contains(keyword) {
                    return "This AppleScript contains \"\(keyword)\" which looks destructive. Are you sure?"
                }
            }
        default:
            break
        }
        return nil
    }

    // MARK: - Helpers

    private func actionDescription(for action: ActionTag) -> String {
        switch action {
        case .click(let x, let y, _):       return "clicking (\(x), \(y))"
        case .doubleClick(let x, let y, _): return "double-clicking (\(x), \(y))"
        case .rightClick(let x, let y, _):  return "right-clicking (\(x), \(y))"
        case .axClick(let label):           return "ax-clicking '\(label)'"
        case .type(let text):
            let preview = text.count > 30 ? String(text.prefix(30)) + "…" : text
            return "typing '\(preview)'"
        case .hotkey(let keys):             return "pressing \(keys)"
        case .scroll(let dir, let amt, _, _, _): return "scrolling \(dir) \(amt)"
        case .appleScript:                  return "running AppleScript"
        case .wait(let ms):                 return "waiting \(ms)ms"
        case .screenshot:                   return "taking screenshot"
        case .confirm(let msg):             return "confirm: \(msg)"
        case .point:                        return "pointing"
        case .taskDone:                     return "task done"
        case .plan(let steps):              return "registering \(steps.count)-step plan"
        case .subtaskDone(let name):        return "marking done: \(name)"
        }
    }

    // MARK: - Virtual key code table

    private func virtualKeyCode(for key: String) -> CGKeyCode? {
        let table: [String: CGKeyCode] = [
            "a": 0,  "s": 1,  "d": 2,  "f": 3,  "h": 4,  "g": 5,  "z": 6,
            "x": 7,  "c": 8,  "v": 9,  "b": 11, "q": 12, "w": 13, "e": 14,
            "r": 15, "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21,
            "6": 22, "5": 23, "=": 24, "9": 25, "7": 26, "-": 27, "8": 28,
            "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35,
            "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43,
            "/": 44, "n": 45, "m": 46, ".": 47, "tab": 48, "space": 49,
            "`": 50, "backspace": 51, "delete": 51, "escape": 53, "esc": 53,
            "return": 36, "enter": 36, "left": 123, "right": 124, "down": 125,
            "up": 126, "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96,
            "f6": 97, "f7": 98, "f8": 100, "f9": 101, "f10": 109,
            "f11": 103, "f12": 111,
        ]
        return table[key.lowercased()]
    }
}
