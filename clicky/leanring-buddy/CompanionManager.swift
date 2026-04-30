//
//  CompanionManager.swift
//  leanring-buddy
//
//  Central state manager for the companion voice mode. Owns the push-to-talk
//  pipeline (dictation manager + global shortcut monitor + overlay) and
//  exposes observable voice state for the panel UI.
//

import AVFoundation
import Combine
import Foundation
import PostHog
import ScreenCaptureKit
import SwiftUI

enum CompanionVoiceState {
    case idle
    case listening
    case processing
    case responding
}

// MARK: - Task Plan (Plan Model)

/// A single step in a multi-step task plan registered via the [PLAN:...] tag.
private struct MickySubtask {
    let name: String
    var isDone: Bool
}

/// Tracks the current task plan so each loop iteration gets an up-to-date
/// view of what's been done and what's still pending. Injected as a Markdown
/// block into the user prompt on every iteration after the plan is registered.
private struct MickyTaskPlan {
    var subtasks: [MickySubtask]

    mutating func markStepDone(matchingName stepName: String) {
        for index in subtasks.indices {
            if subtasks[index].name.lowercased().contains(stepName.lowercased()) {
                subtasks[index].isDone = true
                return
            }
        }
    }

    var markdownBlock: String {
        let lines = subtasks.map { ($0.isDone ? "- [x] " : "- [ ] ") + $0.name }
        return "## Current Task Plan\n" + lines.joined(separator: "\n")
    }
}

@MainActor
final class CompanionManager: ObservableObject {
    @Published private(set) var voiceState: CompanionVoiceState = .idle
    @Published private(set) var lastTranscript: String?
    @Published private(set) var currentAudioPowerLevel: CGFloat = 0
    @Published private(set) var hasAccessibilityPermission = false
    @Published private(set) var hasScreenRecordingPermission = false
    @Published private(set) var hasMicrophonePermission = false
    @Published private(set) var hasScreenContentPermission = false

    /// Screen location (global AppKit coords) of a detected UI element the
    /// buddy should fly to and point at. Parsed from Claude's response;
    /// observed by BlueCursorView to trigger the flight animation.
    @Published var detectedElementScreenLocation: CGPoint?
    /// The display frame (global AppKit coords) of the screen the detected
    /// element is on, so BlueCursorView knows which screen overlay should animate.
    @Published var detectedElementDisplayFrame: CGRect?
    /// Custom speech bubble text for the pointing animation. When set,
    /// BlueCursorView uses this instead of a random pointer phrase.
    @Published var detectedElementBubbleText: String?

    // MARK: - Onboarding Video State (shared across all screen overlays)

    @Published var onboardingVideoPlayer: AVPlayer?
    @Published var showOnboardingVideo: Bool = false
    @Published var onboardingVideoOpacity: Double = 0.0
    private var onboardingVideoEndObserver: NSObjectProtocol?
    private var onboardingDemoTimeObserver: Any?

    // MARK: - Onboarding Prompt Bubble

    /// Text streamed character-by-character on the cursor after the onboarding video ends.
    @Published var onboardingPromptText: String = ""
    @Published var onboardingPromptOpacity: Double = 0.0
    @Published var showOnboardingPrompt: Bool = false

    // MARK: - Onboarding Music

    private var onboardingMusicPlayer: AVAudioPlayer?
    private var onboardingMusicFadeTimer: Timer?

    let buddyDictationManager = BuddyDictationManager()
    let globalPushToTalkShortcutMonitor = GlobalPushToTalkShortcutMonitor()
    let overlayWindowManager = OverlayWindowManager()
    // Response text is now displayed inline on the cursor overlay via
    // streamingResponseText, so no separate response overlay manager is needed.

    /// Base URL for the Cloudflare Worker proxy. All API requests route
    /// through this so keys never ship in the app binary.
    private static let workerBaseURL = "http://localhost:8081"

    private lazy var geminiAPI: GeminiAPI = {
        return GeminiAPI(proxyURL: "\(Self.workerBaseURL)/chat", model: selectedModel)
    }()

    private lazy var geminiTTSClient: GeminiTTSClient = {
        return GeminiTTSClient(proxyURL: "\(Self.workerBaseURL)/tts")
    }()

    /// Conversation history so the model (currently Gemini) remembers prior
    /// exchanges within a session. Each entry is the user's transcript and
    /// the assistant's response. Cleared when the user presses the "new
    /// task" shortcut variant (ctrl + option) — see `handleShortcutTransition`.
    private var conversationHistory: [(userTranscript: String, assistantResponse: String)] = []

    /// The currently running AI response task, if any. Cancelled when the user
    /// speaks again so a new response can begin immediately.
    private var currentResponseTask: Task<Void, Never>?

    private var shortcutTransitionCancellable: AnyCancellable?
    private var voiceStateCancellable: AnyCancellable?
    private var audioPowerCancellable: AnyCancellable?
    private var accessibilityCheckTimer: Timer?
    private var pendingKeyboardShortcutStartTask: Task<Void, Never>?
    /// Scheduled hide for transient cursor mode — cancelled if the user
    /// speaks again before the delay elapses.
    private var transientHideTask: Task<Void, Never>?

    /// True when all three required permissions (accessibility, screen recording,
    /// microphone) are granted. Used by the panel to show a single "all good" state.
    var allPermissionsGranted: Bool {
        hasAccessibilityPermission && hasScreenRecordingPermission && hasMicrophonePermission && hasScreenContentPermission
    }

    /// Whether the blue cursor overlay is currently visible on screen.
    /// Used by the panel to show accurate status text ("Active" vs "Ready").
    @Published private(set) var isOverlayVisible: Bool = false

    /// The Gemini model used for voice responses. Persisted to UserDefaults.
    @Published var selectedModel: String = UserDefaults.standard.string(forKey: "selectedGeminiModel") ?? "gemini-3.1-flash-lite-preview"

    func setSelectedModel(_ model: String) {
        selectedModel = model
        UserDefaults.standard.set(model, forKey: "selectedGeminiModel")
        geminiAPI.model = model
    }

    /// User preference for whether the Micky cursor should be shown.
    /// When toggled off, the overlay is hidden and push-to-talk is disabled.
    /// Persisted to UserDefaults so the choice survives app restarts.
    @Published var isMickyCursorEnabled: Bool = UserDefaults.standard.object(forKey: "isMickyCursorEnabled") == nil
        ? true
        : UserDefaults.standard.bool(forKey: "isMickyCursorEnabled")

    func setMickyCursorEnabled(_ enabled: Bool) {
        isMickyCursorEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: "isMickyCursorEnabled")
        transientHideTask?.cancel()
        transientHideTask = nil

        if enabled {
            overlayWindowManager.hasShownOverlayBefore = true
            overlayWindowManager.showOverlay(onScreens: NSScreen.screens, companionManager: self)
            isOverlayVisible = true
        } else {
            overlayWindowManager.hideOverlay()
            isOverlayVisible = false
        }
    }

    /// Whether the user has completed onboarding at least once. Persisted
    /// to UserDefaults so the Start button only appears on first launch.
    var hasCompletedOnboarding: Bool {
        get { UserDefaults.standard.bool(forKey: "hasCompletedOnboarding") }
        set { UserDefaults.standard.set(newValue, forKey: "hasCompletedOnboarding") }
    }

    /// Whether the user has submitted their email during onboarding.
    @Published var hasSubmittedEmail: Bool = UserDefaults.standard.bool(forKey: "hasSubmittedEmail")

    /// Submits the user's email to FormSpark and identifies them in PostHog.
    func submitEmail(_ email: String) {
        let trimmedEmail = email.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedEmail.isEmpty else { return }

        hasSubmittedEmail = true
        UserDefaults.standard.set(true, forKey: "hasSubmittedEmail")

        // Identify user in PostHog
        PostHogSDK.shared.identify(trimmedEmail, userProperties: [
            "email": trimmedEmail
        ])

        // Submit to FormSpark
        Task {
            var request = URLRequest(url: URL(string: "https://submit-form.com/RWbGJxmIs")!)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try? JSONSerialization.data(withJSONObject: ["email": trimmedEmail])
            _ = try? await URLSession.shared.data(for: request)
        }
    }

    func start() {
        refreshAllPermissions()
        print("🔑 Micky start — accessibility: \(hasAccessibilityPermission), screen: \(hasScreenRecordingPermission), mic: \(hasMicrophonePermission), screenContent: \(hasScreenContentPermission), onboarded: \(hasCompletedOnboarding)")
        startPermissionPolling()
        bindVoiceStateObservation()
        bindAudioPowerLevel()
        bindShortcutTransitions()
        // Eagerly touch the Gemini API so its TLS warmup handshake completes
        // well before the onboarding demo fires at ~40s into the video.
        _ = geminiAPI

        // If the user already completed onboarding AND all permissions are
        // still granted, show the cursor overlay immediately. If permissions
        // were revoked (e.g. signing change), don't show the cursor — the
        // panel will show the permissions UI instead.
        if hasCompletedOnboarding && allPermissionsGranted && isMickyCursorEnabled {
            overlayWindowManager.hasShownOverlayBefore = true
            overlayWindowManager.showOverlay(onScreens: NSScreen.screens, companionManager: self)
            isOverlayVisible = true
        }
    }

    /// Called by BlueCursorView after the buddy finishes its pointing
    /// animation and returns to cursor-following mode.
    /// Triggers the onboarding sequence — dismisses the panel and restarts
    /// the overlay so the welcome animation and intro video play.
    func triggerOnboarding() {
        // Post notification so the panel manager can dismiss the panel
        NotificationCenter.default.post(name: .mickyDismissPanel, object: nil)

        // Mark onboarding as completed so the Start button won't appear
        // again on future launches — the cursor will auto-show instead
        hasCompletedOnboarding = true

        MickyAnalytics.trackOnboardingStarted()

        // Play Besaid theme at 60% volume, fade out after 1m 30s
        startOnboardingMusic()

        // Show the overlay for the first time — isFirstAppearance triggers
        // the welcome animation and onboarding video
        overlayWindowManager.showOverlay(onScreens: NSScreen.screens, companionManager: self)
        isOverlayVisible = true
    }

    /// Replays the onboarding experience from the "Watch Onboarding Again"
    /// footer link. Same flow as triggerOnboarding but the cursor overlay
    /// is already visible so we just restart the welcome animation and video.
    func replayOnboarding() {
        NotificationCenter.default.post(name: .mickyDismissPanel, object: nil)
        MickyAnalytics.trackOnboardingReplayed()
        startOnboardingMusic()
        // Tear down any existing overlays and recreate with isFirstAppearance = true
        overlayWindowManager.hasShownOverlayBefore = false
        overlayWindowManager.showOverlay(onScreens: NSScreen.screens, companionManager: self)
        isOverlayVisible = true
    }

    private func stopOnboardingMusic() {
        onboardingMusicFadeTimer?.invalidate()
        onboardingMusicFadeTimer = nil
        onboardingMusicPlayer?.stop()
        onboardingMusicPlayer = nil
    }

    private func startOnboardingMusic() {
        stopOnboardingMusic()
        guard let musicURL = Bundle.main.url(forResource: "ff", withExtension: "mp3") else {
            print("⚠️ Micky: ff.mp3 not found in bundle")
            return
        }

        do {
            let player = try AVAudioPlayer(contentsOf: musicURL)
            player.volume = 0.3
            player.play()
            self.onboardingMusicPlayer = player

            // After 1m 30s, fade the music out over 3s
            onboardingMusicFadeTimer = Timer.scheduledTimer(withTimeInterval: 90.0, repeats: false) { [weak self] _ in
                self?.fadeOutOnboardingMusic()
            }
        } catch {
            print("⚠️ Micky: Failed to play onboarding music: \(error)")
        }
    }

    private func fadeOutOnboardingMusic() {
        guard let player = onboardingMusicPlayer else { return }

        let fadeSteps = 30
        let fadeDuration: Double = 3.0
        let stepInterval = fadeDuration / Double(fadeSteps)
        let volumeDecrement = player.volume / Float(fadeSteps)
        var stepsRemaining = fadeSteps

        onboardingMusicFadeTimer = Timer.scheduledTimer(withTimeInterval: stepInterval, repeats: true) { [weak self] timer in
            stepsRemaining -= 1
            player.volume -= volumeDecrement

            if stepsRemaining <= 0 {
                timer.invalidate()
                player.stop()
                self?.onboardingMusicPlayer = nil
                self?.onboardingMusicFadeTimer = nil
            }
        }
    }

    func clearDetectedElementLocation() {
        detectedElementScreenLocation = nil
        detectedElementDisplayFrame = nil
        detectedElementBubbleText = nil
    }

    func stop() {
        globalPushToTalkShortcutMonitor.stop()
        buddyDictationManager.cancelCurrentDictation()
        overlayWindowManager.hideOverlay()
        transientHideTask?.cancel()

        currentResponseTask?.cancel()
        currentResponseTask = nil
        shortcutTransitionCancellable?.cancel()
        voiceStateCancellable?.cancel()
        audioPowerCancellable?.cancel()
        accessibilityCheckTimer?.invalidate()
        accessibilityCheckTimer = nil
    }

    func refreshAllPermissions() {
        let previouslyHadAccessibility = hasAccessibilityPermission
        let previouslyHadScreenRecording = hasScreenRecordingPermission
        let previouslyHadMicrophone = hasMicrophonePermission
        let previouslyHadAll = allPermissionsGranted

        let currentlyHasAccessibility = WindowPositionManager.hasAccessibilityPermission()
        hasAccessibilityPermission = currentlyHasAccessibility

        if currentlyHasAccessibility {
            globalPushToTalkShortcutMonitor.start()
        } else {
            globalPushToTalkShortcutMonitor.stop()
        }

        // CGPreflightScreenCaptureAccess() returns false until the app is restarted after
        // being granted. Use the "previously confirmed" fallback so the UI shows Granted
        // immediately after the user approves in System Settings without needing a relaunch.
        hasScreenRecordingPermission = WindowPositionManager.shouldTreatScreenRecordingPermissionAsGrantedForSessionLaunch()

        let micAuthStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        hasMicrophonePermission = micAuthStatus == .authorized

        // Debug: log permission state on changes
        if previouslyHadAccessibility != hasAccessibilityPermission
            || previouslyHadScreenRecording != hasScreenRecordingPermission
            || previouslyHadMicrophone != hasMicrophonePermission {
            print("🔑 Permissions — accessibility: \(hasAccessibilityPermission), screen: \(hasScreenRecordingPermission), mic: \(hasMicrophonePermission), screenContent: \(hasScreenContentPermission)")
        }

        // Track individual permission grants as they happen
        if !previouslyHadAccessibility && hasAccessibilityPermission {
            MickyAnalytics.trackPermissionGranted(permission: "accessibility")
        }
        if !previouslyHadScreenRecording && hasScreenRecordingPermission {
            MickyAnalytics.trackPermissionGranted(permission: "screen_recording")
        }
        if !previouslyHadMicrophone && hasMicrophonePermission {
            MickyAnalytics.trackPermissionGranted(permission: "microphone")
        }
        // Screen content permission is persisted — once the user has approved the
        // SCShareableContent picker, we don't need to re-check it.
        if !hasScreenContentPermission {
            hasScreenContentPermission = UserDefaults.standard.bool(forKey: "hasScreenContentPermission")
        }

        if !previouslyHadAll && allPermissionsGranted {
            MickyAnalytics.trackAllPermissionsGranted()
        }
    }

    /// Triggers the macOS screen content picker by performing a dummy
    /// screenshot capture. Once the user approves, we persist the grant
    /// so they're never asked again during onboarding.
    @Published private(set) var isRequestingScreenContent = false

    func requestScreenContentPermission() {
        guard !isRequestingScreenContent else { return }
        isRequestingScreenContent = true
        Task {
            do {
                let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
                guard let display = content.displays.first else {
                    await MainActor.run { isRequestingScreenContent = false }
                    return
                }
                let filter = SCContentFilter(display: display, excludingWindows: [])
                let config = SCStreamConfiguration()
                config.width = 320
                config.height = 240
                let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
                // Verify the capture actually returned real content — a 0x0 or
                // fully-empty image means the user denied the prompt.
                let didCapture = image.width > 0 && image.height > 0
                print("🔑 Screen content capture result — width: \(image.width), height: \(image.height), didCapture: \(didCapture)")
                await MainActor.run {
                    isRequestingScreenContent = false
                    guard didCapture else { return }
                    hasScreenContentPermission = true
                    UserDefaults.standard.set(true, forKey: "hasScreenContentPermission")
                    MickyAnalytics.trackPermissionGranted(permission: "screen_content")

                    // If onboarding was already completed, show the cursor overlay now
                    if hasCompletedOnboarding && allPermissionsGranted && !isOverlayVisible && isMickyCursorEnabled {
                        overlayWindowManager.hasShownOverlayBefore = true
                        overlayWindowManager.showOverlay(onScreens: NSScreen.screens, companionManager: self)
                        isOverlayVisible = true
                    }
                }
            } catch {
                print("⚠️ Screen content permission request failed: \(error)")
                await MainActor.run { isRequestingScreenContent = false }
            }
        }
    }

    // MARK: - Private

    /// Triggers the system microphone prompt if the user has never been asked.
    /// Once granted/denied the status sticks and polling picks it up.
    private func promptForMicrophoneIfNotDetermined() {
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .notDetermined else { return }
        AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
            Task { @MainActor [weak self] in
                self?.hasMicrophonePermission = granted
            }
        }
    }

    /// Polls all permissions frequently so the UI updates live after the
    /// user grants them in System Settings. Screen Recording is the exception —
    /// macOS requires an app restart for that one to take effect.
    private func startPermissionPolling() {
        accessibilityCheckTimer = Timer.scheduledTimer(withTimeInterval: 1.5, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.refreshAllPermissions()
            }
        }
    }

    private func bindAudioPowerLevel() {
        audioPowerCancellable = buddyDictationManager.$currentAudioPowerLevel
            .receive(on: DispatchQueue.main)
            .sink { [weak self] powerLevel in
                self?.currentAudioPowerLevel = powerLevel
            }
    }

    private func bindVoiceStateObservation() {
        voiceStateCancellable = buddyDictationManager.$isRecordingFromKeyboardShortcut
            .combineLatest(
                buddyDictationManager.$isFinalizingTranscript,
                buddyDictationManager.$isPreparingToRecord
            )
            .receive(on: DispatchQueue.main)
            .sink { [weak self] isRecording, isFinalizing, isPreparing in
                guard let self else { return }
                // Don't override .responding — the AI response pipeline
                // manages that state directly until streaming finishes.
                guard self.voiceState != .responding else { return }

                if isFinalizing {
                    self.voiceState = .processing
                } else if isRecording {
                    self.voiceState = .listening
                } else if isPreparing {
                    self.voiceState = .processing
                } else {
                    self.voiceState = .idle
                    // If the user pressed and released the hotkey without
                    // saying anything, no response task runs — schedule the
                    // transient hide here so the overlay doesn't get stuck.
                    // Only do this when no response is in flight, otherwise
                    // the brief idle gap between recording and processing
                    // would prematurely hide the overlay.
                    if self.currentResponseTask == nil {
                        self.scheduleTransientHideIfNeeded()
                    }
                }
            }
    }

    private func bindShortcutTransitions() {
        shortcutTransitionCancellable = globalPushToTalkShortcutMonitor
            .shortcutTransitionPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] transition in
                self?.handleShortcutTransition(transition)
            }
    }

    private func handleShortcutTransition(_ transition: BuddyPushToTalkShortcut.ShortcutTransition) {
        switch transition {
        case .pressed(let isNewTask):
            guard !buddyDictationManager.isDictationInProgress else { return }
            // Don't register push-to-talk while the onboarding video is playing
            guard !showOnboardingVideo else { return }

            // If the user pressed the "new task" variant (ctrl + option),
            // wipe the running conversation history before starting the
            // recording. The "continue task" variant (cmd + option)
            // preserves history so multi-turn corrections (e.g. "you sent
            // it to the wrong person") share full context. The clear runs
            // synchronously here so the dictation start that follows can
            // never race with a stale-history send.
            if isNewTask && !conversationHistory.isEmpty {
                let clearedTurnCount = conversationHistory.count
                conversationHistory.removeAll()
                print("🧹 New-task shortcut (ctrl+option) — cleared \(clearedTurnCount) prior turn(s) from conversation history")
            } else if !isNewTask {
                print("🔁 Continue-task shortcut (cmd+option) — preserving \(conversationHistory.count) prior turn(s)")
            }

            // Cancel any pending transient hide so the overlay stays visible
            transientHideTask?.cancel()
            transientHideTask = nil

            // If the cursor is hidden, bring it back transiently for this interaction
            if !isMickyCursorEnabled && !isOverlayVisible {
                overlayWindowManager.hasShownOverlayBefore = true
                overlayWindowManager.showOverlay(onScreens: NSScreen.screens, companionManager: self)
                isOverlayVisible = true
            }

            // Dismiss the menu bar panel so it doesn't cover the screen
            NotificationCenter.default.post(name: .mickyDismissPanel, object: nil)

            // Cancel any in-progress response and TTS from a previous utterance
            currentResponseTask?.cancel()
            geminiTTSClient.stopPlayback()
            clearDetectedElementLocation()

            // Dismiss the onboarding prompt if it's showing
            if showOnboardingPrompt {
                withAnimation(.easeOut(duration: 0.3)) {
                    onboardingPromptOpacity = 0.0
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                    self.showOnboardingPrompt = false
                    self.onboardingPromptText = ""
                }
            }
    

            MickyAnalytics.trackPushToTalkStarted()

            pendingKeyboardShortcutStartTask?.cancel()
            pendingKeyboardShortcutStartTask = Task {
                await buddyDictationManager.startPushToTalkFromKeyboardShortcut(
                    currentDraftText: "",
                    updateDraftText: { _ in
                        // Partial transcripts are hidden (waveform-only UI)
                    },
                    submitDraftText: { [weak self] finalTranscript in
                        self?.lastTranscript = finalTranscript
                        print("🗣️ Companion received transcript: \(finalTranscript)")
                        MickyAnalytics.trackUserMessageSent(transcript: finalTranscript)
                        self?.sendTranscriptToClaudeWithScreenshot(transcript: finalTranscript)
                    }
                )
            }
        case .released:
            // Cancel the pending start task in case the user released the shortcut
            // before the async startPushToTalk had a chance to begin recording.
            // Without this, a quick press-and-release drops the release event and
            // leaves the waveform overlay stuck on screen indefinitely.
            MickyAnalytics.trackPushToTalkReleased()
            pendingKeyboardShortcutStartTask?.cancel()
            pendingKeyboardShortcutStartTask = nil
            buddyDictationManager.stopPushToTalkFromKeyboardShortcut()
        case .none:
            break
        }
    }

    // MARK: - Companion Prompt

    private static var companionVoiceResponseSystemPrompt: String {
        let realUserHome = NSHomeDirectoryForUser(NSUserName()) ?? NSHomeDirectory()
        let contextFile = URL(fileURLWithPath: realUserHome).appendingPathComponent("Downloads/varun_agent/micky_context.md")
        let contextSection: String
        if let context = try? String(contentsOf: contextFile) {
            contextSection = "\n\n── PERSONAL CONTEXT ──\n\(context)\n"
        } else {
            contextSection = ""
        }
        return buildSystemPrompt()
            + installedApplicationsSection()
            + laptopFoldersSection()
            + contextSection
    }

    /// Reads `~/laptop_wiki/index.md` and returns a section listing the user's
    /// notable folders with one-line purpose summaries. The wiki is the source
    /// of truth for "where does folder X live on this Mac" — the indexer keeps
    /// it in sync with the filesystem, so the model can rely on these absolute
    /// paths instead of guessing or running `find` for every folder reference.
    /// Re-read on every prompt build (file is small) so wiki updates are picked
    /// up without restarting the app.
    private static func laptopFoldersSection() -> String {
        let realUserHome = NSHomeDirectoryForUser(NSUserName()) ?? NSHomeDirectory()
        let indexFilePath = "\(realUserHome)/laptop_wiki/index.md"
        guard let rawIndex = try? String(contentsOfFile: indexFilePath, encoding: .utf8) else {
            return ""
        }

        // index.md has rows like:
        //   | [/Users/varuntyagi/Downloads/kincare](folders/...md) | Purpose text. |
        // Pull (path, purpose) from each table row.
        var folderEntries: [(absolutePath: String, purpose: String)] = []
        for line in rawIndex.components(separatedBy: "\n") {
            guard line.hasPrefix("| [") else { continue }
            guard let pathOpenBracket = line.range(of: "["),
                  let pathCloseBracket = line.range(of: "](", range: pathOpenBracket.upperBound..<line.endIndex),
                  let purposeSeparator = line.range(of: " | ", range: pathCloseBracket.upperBound..<line.endIndex) else {
                continue
            }
            let absolutePath = String(line[pathOpenBracket.upperBound..<pathCloseBracket.lowerBound])
            var purpose = String(line[purposeSeparator.upperBound..<line.endIndex])
            if purpose.hasSuffix(" |") { purpose = String(purpose.dropLast(2)) }
            folderEntries.append((absolutePath, purpose.trimmingCharacters(in: .whitespaces)))
        }

        guard !folderEntries.isEmpty else { return "" }

        let formattedEntries = folderEntries
            .map { "  \($0.absolutePath) — \($0.purpose)" }
            .joined(separator: "\n")

        return """


        ── KNOWN FOLDERS ON THIS MAC ──
        the user's notable folders, with one-line summaries (sourced from ~/laptop_wiki/index.md, regenerated when the user adds or moves files). when the user mentions a folder by partial or casual name (e.g. "open the kincare folder", "open new media"), match it against this list FIRST to find the real absolute path. only fall back to `find` if no entry below plausibly matches. always pass absolute paths to shell commands — never `~`.

        \(formattedEntries)
        """
    }


    /// Lists the apps actually installed on this Mac so the model has a closed
    /// set to draw from when constructing `open -a 'NAME'` calls. The list is
    /// dynamic — whatever the user has on disk — there is no hardcoded alias
    /// table. The executor additionally fuzzy-matches whatever the model writes
    /// against this same set, so a near-miss still resolves correctly.
    private static func installedApplicationsSection() -> String {
        let installedAppsList = InstalledApplicationCatalog.allInstalledAppNames.joined(separator: ", ")
        return """


        ── INSTALLED APPS ──
        these are the apps actually installed on this machine. when you write `open -a 'NAME'`, prefer one of these exact names. you may write a casual short form (the executor fuzzy-matches it back to a real bundle name), but using a name from this list is faster and more reliable. if the user asks for an app that isn't on this list, tell them it's not installed instead of guessing.

        \(installedAppsList)
        """
    }

    private static func buildSystemPrompt() -> String { return """
    you're micky, a friendly always-on companion that lives in the user's menu bar. the user just spoke to you via push-to-talk and you can see their screen(s). your reply will be spoken aloud via text-to-speech, so write the way you'd actually talk. this is an ongoing conversation — you remember everything they've said before.

    rules:
    - default to one or two sentences. be direct and dense. BUT if the user asks you to explain more, go deeper, or elaborate, then go all out.
    - all lowercase, casual, warm. no emojis.
    - write for the ear, not the eye. short sentences. no lists, bullet points, markdown, or formatting — just natural speech.
    - don't use abbreviations or symbols that sound weird read aloud. write "for example" not "e.g.", spell out small numbers.
    - if the user's question relates to what's on their screen, reference specific things you see.
    - never say "simply" or "just".
    - don't read out code verbatim. describe what it does conversationally.
    - if you receive multiple screen images, the one labeled "primary focus" is where the cursor is.

    ── ACTIONS ──
    you can actually DO things on the user's computer, not just talk about them. when the user asks you to open something, click something, type something, or automate anything — do it using action tags.

    action tags are embedded inline in your response. the user's spoken text is read aloud; the tags are executed silently. always say what you're about to do in one short sentence, then include the tags.

    available action tags:

    [AXCLICK:label]            — PREFERRED click method. clicks a UI element by its visible text or accessible label (e.g. [AXCLICK:Send], [AXCLICK:Canara NSUT]). walks the AX tree first, falls back to Vision OCR. use this instead of [CLICK] whenever the target has visible text.
    [CLICK:x,y]               — LAST RESORT pixel click. only use when the target has no accessible label.
    [CLICK:x,y:screenN]       — pixel click on screen N
    [DBLCLICK:x,y]            — double-click at pixel coordinates
    [RCLICK:x,y]              — right-click at pixel coordinates
    [TYPE:text]                — type text into the focused field. use \\] to include a literal ]
    [HOTKEY:cmd+space]        — press a keyboard shortcut. modifiers: cmd, shift, option, ctrl
    [SCROLL:down:3:x,y]       — scroll down 3 lines at x,y. directions: up/down/left/right
    [APPLESCRIPT:source]      — run AppleScript. best for opening apps, sending messages, file ops
    [WAIT:500]                 — wait 500 milliseconds before next action
    [SCREENSHOT]               — take a fresh screenshot and re-evaluate (use when waiting for UI to load)
    [CONFIRM:message]          — ask user to confirm before continuing (use for destructive actions)
    [POINT:x,y:label]          — point the blue cursor at something (visual only, not an action)
    [POINT:none]               — don't point
    [TASK_DONE]                — REQUIRED at end of every completed task. stops the agentic loop. always emit this with a spoken summary when the task is fully done.
    [PLAN:step1|step2|step3]   — register a multi-step plan at the start of complex tasks. micky tracks each step and shows you the current state on every turn so you stay oriented. example: [PLAN:open whatsapp|find contact|type message|send]
    [SUBTASK_DONE:step name]   — mark a plan step complete when you finish it. example: [SUBTASK_DONE:find contact]

    examples:

    user: "open VS Code and open the kincare folder"
    response: on it, opening the kincare folder in VS Code.
    [APPLESCRIPT:do shell script "open -a 'Visual Studio Code - Insiders' '/Users/varuntyagi/Downloads/kincare'"]

    user: "open whatsapp and send hi to canara"
    response: opening WhatsApp and sending hi to canara.
    [APPLESCRIPT:do shell script "open -a WhatsApp"]
    [WAIT:3000]
    [SCREENSHOT]
    (after screenshot: use the search bar at the top of WhatsApp to search for the contact — this is more reliable than scrolling. click the search bar, type the contact name, wait for results, then click the contact.)
    example after screenshot:
    [CLICK:600,60]
    [WAIT:500]
    [TYPE:canara]
    [WAIT:1000]
    [SCREENSHOT]
    (after search screenshot: identify the correct result row — match by name, prefer "Chats" results over "Groups" unless the user said "group". click the row, then take ANOTHER screenshot to verify the conversation header shows the right contact's name BEFORE you type. never type or press return until that verification screenshot confirms the correct conversation is open.)
    [CLICK:600,200]
    [WAIT:800]
    [SCREENSHOT]
    (verify the conversation header on the right side now shows "canara". if it shows a different name or a group, stop and tell the user. if it matches, proceed:)
    [TYPE:hi]
    [HOTKEY:return]

    user: "open that pdf in preview"
    response: opening it in preview now.
    [APPLESCRIPT:do shell script "open -a Preview '/Users/varuntyagi/path/to/file.pdf'"]

    user: "open the downloaded movie in VLC"
    response: let me find it and open it in VLC.
    [APPLESCRIPT:do shell script "open -a VLC '/Users/varuntyagi/Downloads/movie.mp4'"]

    user: "open safari"
    response: opening safari now.
    [APPLESCRIPT:do shell script "open -a Safari"]

    user: "open terminal"
    response: opening terminal.
    [APPLESCRIPT:do shell script "open -a Terminal"]

    user: "open spotlight and search for terminal"
    response: opening spotlight now.
    [HOTKEY:cmd+space]
    [WAIT:300]
    [TYPE:terminal]
    [HOTKEY:return]

    rules for actions:
    - for clicking UI elements with visible text or accessible labels: ALWAYS use [AXCLICK:label] instead of [CLICK:x,y]. the executor walks the accessibility tree and finds the actual element location — no coordinate guessing. example: [AXCLICK:Send] to click a Send button, [AXCLICK:Canara NSUT] to open that chat row. if [AXCLICK] reports it can't find the element, the executor returns the AX labels and OCR text it DID see — use that to pick a better label, not to fall back to pixel clicks.
    - for multi-step tasks (more than 3 actions), register a plan first: [PLAN:step one|step two|step three]. micky will show you the plan state on every turn so you can track progress and know which step you're on. mark each step done with [SUBTASK_DONE:step name] as you complete it.
    - ALWAYS emit [TASK_DONE] with a spoken summary when the task is fully complete. the agentic loop keeps running until it sees [TASK_DONE] — never leave the loop running after the task is done.
    - to open any app, ALWAYS use: [APPLESCRIPT:do shell script "open -a AppName"] — NEVER use "tell application X to activate", it doesn't work reliably
    - in shell scripts, ALWAYS use absolute paths starting with /Users/varuntyagi/... never use ~ — `~` may resolve to a sandbox container path on some builds and silently fail
    - CRITICAL: never emit [TYPE] or [HOTKEY:return] (or any text/keystroke that sends a message, runs a command, or commits an input) immediately after a [CLICK] that opens a different view, contact, document, or chat. you MUST insert a [SCREENSHOT] between the click and the type, AND visually confirm from that screenshot that the click landed on the right element. this is what prevents wrong-recipient messages, wrong-file edits, and wrong-form submissions. if the verification screenshot shows the wrong target, stop and tell the user instead of proceeding.
    - the rule above applies even if you "know" the click was correct from the previous screenshot — UIs can move, scroll, or reflow between when you saw them and when your click landed. always verify.
    - use [SCREENSHOT] after app launches, after every state-changing click (search results, contact rows, list items, dropdowns, dialogs), and any time the next action depends on the previous one having succeeded
    - use [WAIT:2000][SCREENSHOT] after opening an app — give it time to load before looking at the screen
    - max 20 action tags per turn. if a task needs more, use [SCREENSHOT] to re-evaluate mid-task
    - use [CONFIRM] before anything that deletes files or sends a message to a recipient whose identity you weren't able to fully verify in a screenshot
    - to resolve a folder the user mentions: FIRST scan the KNOWN FOLDERS section below for a match (this is the wiki of their notable directories with absolute paths). If you find a match, use that absolute path directly. ONLY if no entry there reasonably matches, fall back to: [APPLESCRIPT:do shell script "find /Users/varuntyagi/Downloads /Users/varuntyagi/Desktop /Users/varuntyagi/Documents -name '*keyword*' 2>/dev/null | head -5"]
    - for WhatsApp messages: if the contact's phone number is in the personal context above, use the URL scheme directly: [APPLESCRIPT:do shell script "open 'whatsapp://send?phone=PHONENUMBER&text=MESSAGE'"] — this is the most reliable method, no clicking needed and no risk of wrong-recipient
    - if phone number is not known, fall back to: open WhatsApp → wait 3000ms → screenshot → click search bar → type name → screenshot → click the "Chats" result (NOT Groups, unless the user asked for the group) → wait 800ms → SCREENSHOT and verify the conversation header shows the correct contact name → type message → return
    - app names: pick from the INSTALLED APPS list below. the user may say a partial or casual name; you can write that casual name in `open -a` and the executor fuzzy-matches it against installed apps, but you'll be faster and more reliable if you use the canonical name from the list directly. if no installed app reasonably matches what the user asked for, say so — don't invent.
    - do NOT use System Events keystroke — use [TYPE] and [HOTKEY] instead, they work with any focused app
    - always say what you're doing in natural speech before the action tags

    ── POINTING ──
    use [POINT:x,y:label] to visually point the blue cursor at things on screen. coordinates are in screenshot pixel space (top-left origin). append :screenN if the element is on a different screen.
    """ }  // end buildSystemPrompt

    // MARK: - AI Response Pipeline

    private let agenticExecutor = AgenticActionExecutor()

    /// Agentic loop: captures screens → asks Gemini → speaks response →
    /// executes action tags → if [SCREENSHOT] tag found, loops again with
    /// fresh screen context. Hard cap: 5 iterations per turn.
    private func sendTranscriptToClaudeWithScreenshot(transcript: String) {
        currentResponseTask?.cancel()
        geminiTTSClient.stopPlayback()

        currentResponseTask = Task {
            voiceState = .processing

            var currentUserPrompt = transcript
            var iterationCount = 0
            let maxIterations = 8
            var allSpokenText: [String] = []
            var currentTaskPlan: MickyTaskPlan? = nil

            // Save a placeholder history entry up front so any interrupt
            // (cmd+option mid-loop, cancellation, error) still preserves the
            // user's transcript and whatever assistant text has accumulated.
            // We update entry index `historyEntryIndex` after each iteration.
            conversationHistory.append((userTranscript: transcript, assistantResponse: ""))
            let historyEntryIndex = conversationHistory.count - 1
            if conversationHistory.count > 10 {
                conversationHistory.removeFirst(conversationHistory.count - 10)
            }
            // Index may have shifted if we just trimmed the history.
            let stableHistoryIndex = min(historyEntryIndex, conversationHistory.count - 1)

            do {
                while iterationCount < maxIterations {
                    guard !Task.isCancelled else { return }
                    iterationCount += 1

                    let screenCaptures = try await CompanionScreenCaptureUtility.captureAllScreensAsJPEG()
                    guard !Task.isCancelled else { return }

                    let labeledImages = screenCaptures.map { capture in
                        let dimensionInfo = " (image dimensions: \(capture.screenshotWidthInPixels)x\(capture.screenshotHeightInPixels) pixels)"
                        print("📸 Screenshot: \(capture.screenshotWidthInPixels)x\(capture.screenshotHeightInPixels)px — \(capture.label)")
                        return (data: capture.imageData, label: capture.label + dimensionInfo)
                    }

                    // Tell the executor the real screenshot dimensions so clicks land correctly.
                    // Use the cursor screen's capture — that's always screenCaptures[0] since
                    // the cursor screen is listed first by captureAllScreensAsJPEG.
                    if let primaryCapture = screenCaptures.first {
                        agenticExecutor.lastScreenshotWidth     = primaryCapture.screenshotWidthInPixels
                        agenticExecutor.lastScreenshotHeight    = primaryCapture.screenshotHeightInPixels
                        agenticExecutor.lastScreenshotImageData = primaryCapture.imageData
                    }

                    // Compact old history turns into a summary before they overflow the context window.
                    compactConversationHistoryIfNeeded()

                    let historyForAPI = conversationHistory.map { entry in
                        (userPlaceholder: entry.userTranscript, assistantResponse: entry.assistantResponse)
                    }

                    // Build the per-iteration system prompt: base + relevant memories on first turn only.
                    let iterationSystemPrompt: String
                    if iterationCount == 1 {
                        let memoryBlock = MickyMemoryStore.shared.relevantMemoriesSystemBlock(for: transcript)
                        iterationSystemPrompt = Self.companionVoiceResponseSystemPrompt + memoryBlock
                    } else {
                        iterationSystemPrompt = Self.companionVoiceResponseSystemPrompt
                    }

                    // Build the per-iteration user prompt: plan block (if registered) + the actual prompt.
                    var promptForThisTurn = currentUserPrompt
                    if let plan = currentTaskPlan {
                        promptForThisTurn = plan.markdownBlock + "\n\n" + promptForThisTurn
                    }

                    let (fullResponseText, _) = try await geminiAPI.analyzeImageStreaming(
                        images: labeledImages,
                        systemPrompt: iterationSystemPrompt,
                        conversationHistory: historyForAPI,
                        userPrompt: promptForThisTurn,
                        onTextChunk: { _ in }
                    )

                    guard !Task.isCancelled else { return }

                    print("🤖 Gemini response: \(fullResponseText.prefix(300))")

                    // Parse all action tags from the response
                    let parsed = AgenticTagParser.parse(fullResponseText)
                    let spokenText = parsed.spokenText
                    print("🎬 Actions parsed: \(parsed.actions.count) — spoken: \(spokenText.prefix(100))")
                    allSpokenText.append(spokenText)

                    // Handle [PLAN] tag — register a task plan on first occurrence
                    if currentTaskPlan == nil,
                       let planAction = parsed.actions.first(where: { if case .plan = $0 { return true }; return false }),
                       case .plan(let steps) = planAction {
                        currentTaskPlan = MickyTaskPlan(
                            subtasks: steps.map { MickySubtask(name: $0, isDone: false) }
                        )
                        print("📋 Task plan registered: \(steps.joined(separator: " → "))")
                    }

                    // Handle [SUBTASK_DONE] tags — advance plan progress
                    for action in parsed.actions {
                        if case .subtaskDone(let stepName) = action {
                            currentTaskPlan?.markStepDone(matchingName: stepName)
                            print("📋 Subtask marked done: \(stepName)")
                        }
                    }

                    // Handle [POINT] tag for the visual cursor animation
                    if let pointTag = parsed.pointTag,
                       case .point(let px, let py, _, let screenNum) = pointTag,
                       px > 0 || py > 0 {
                        applyPointingCoordinates(
                            x: px, y: py,
                            screenNumber: screenNum,
                            screenCaptures: screenCaptures,
                            label: { if case .point(_, _, let lbl, _) = pointTag { return lbl }; return nil }()
                        )
                        voiceState = .idle
                    }

                    // Speak the response
                    let trimmedSpokenText = spokenText.trimmingCharacters(in: .whitespacesAndNewlines)
                    print("🔈 TTS (\(trimmedSpokenText.count) chars): \(trimmedSpokenText.prefix(100))")
                    // Update history eagerly so an interrupt mid-loop still
                    // preserves what was said this turn. allSpokenText was
                    // already appended above when parsed.spokenText was added.
                    if stableHistoryIndex < conversationHistory.count {
                        conversationHistory[stableHistoryIndex] = (
                            userTranscript: transcript,
                            assistantResponse: allSpokenText.joined(separator: " ")
                        )
                    }

                    // Check if Gemini declared the task complete — needs to be
                    // computed before TTS so we know whether to await playback.
                    let isTaskDone = parsed.actions.contains {
                        if case .taskDone = $0 { return true }
                        return false
                    }

                    // TTS strategy: on the final turn (task done) we await playback
                    // so the user hears the full summary. On intermediate turns we
                    // fire-and-forget so the next API call + action execution can
                    // overlap with audio playback — this is the single biggest win
                    // for end-to-end latency on multi-iteration tasks.
                    if !trimmedSpokenText.isEmpty {
                        if isTaskDone {
                            do {
                                try await geminiTTSClient.speakText(trimmedSpokenText)
                                voiceState = .responding
                            } catch {
                                MickyAnalytics.trackTTSError(error: error.localizedDescription)
                                print("⚠️ Gemini TTS error: \(error)")
                            }
                        } else {
                            voiceState = .responding
                            Task { [weak self] in
                                do {
                                    try await self?.geminiTTSClient.speakText(trimmedSpokenText)
                                } catch {
                                    MickyAnalytics.trackTTSError(error: error.localizedDescription)
                                    print("⚠️ Gemini TTS error (background): \(error)")
                                }
                            }
                        }
                    }

                    // Execute action tags that have side effects.
                    // point, taskDone, plan, subtaskDone are handled above
                    // at CompanionManager level and should not go to the executor.
                    let executableActions = parsed.actions.filter {
                        if case .point = $0 { return false }
                        if case .taskDone = $0 { return false }
                        if case .plan = $0 { return false }
                        if case .subtaskDone = $0 { return false }
                        return true
                    }

                    var executedActionSummary = "actions completed"

                    if !executableActions.isEmpty {
                        let outcome = await agenticExecutor.execute(
                            actions: executableActions,
                            onActionStart: { description in
                                print("⚡ Action: \(description)")
                            },
                            onActionFinish: {}
                        )

                        switch outcome {
                        case .needsScreenshotRefresh(let executedSummary):
                            // Either a [SCREENSHOT] tag was reached (normal verification),
                            // OR the executor refused an unverified destructive action
                            // and is forcing a screenshot for re-plan. The "BLOCKED:"
                            // marker (set by AgenticActionExecutor.unverifiedClickBlockReason)
                            // distinguishes them — when present, the user-prompt text
                            // tells Gemini what was refused and why so it can re-plan
                            // instead of blindly retrying the same unsafe sequence.
                            if executedSummary.contains("BLOCKED:") {
                                currentUserPrompt = "[fresh screenshot attached. the executor REFUSED a destructive action because the prior click hadn't been verified. details: \(executedSummary). DO NOT immediately re-emit the blocked TYPE or HOTKEY:return — first look at this screenshot, confirm whether the prior click landed on the right target (correct chat header, right form field, right document, etc.). if it did, proceed with the destructive action. if it didn't, click the correct target FIRST, then take ANOTHER [SCREENSHOT], then type. original task: \(transcript).]"
                            } else {
                                currentUserPrompt = "[fresh screenshot attached. actions just executed: \(executedSummary). continue working on the original task: \(transcript). if the expected app/UI is not visible, try a different approach. emit [TASK_DONE] when the task is fully complete.]"
                            }
                            continue
                        case .cancelled:
                            voiceState = .idle
                            scheduleTransientHideIfNeeded()
                            return
                        case .blockedAwaitingConfirmation(let message, _):
                            // Speak the confirmation request
                            let confirmMsg = "hold on — \(message)"
                            try? await geminiTTSClient.speakText(confirmMsg)
                            voiceState = .idle
                            scheduleTransientHideIfNeeded()
                            return
                        case .completed:
                            executedActionSummary = "actions completed successfully"
                        }
                    }

                    // Decide whether to loop or stop
                    if isTaskDone {
                        // Store a procedural memory of this completed task so future
                        // similar tasks can benefit from what approach worked.
                        let taskSummary = String(allSpokenText.joined(separator: " ").prefix(400))
                        let taskKeywords = extractKeywordsForMemory(from: transcript)
                        if !taskSummary.isEmpty && !taskKeywords.isEmpty {
                            MickyMemoryStore.shared.store(
                                category: .procedural,
                                content: "Task: \(transcript.prefix(100)). Outcome: \(taskSummary)",
                                keywords: taskKeywords
                            )
                        }
                        // Gemini explicitly declared done — exit the loop
                        break
                    } else if executableActions.isEmpty {
                        // Gemini narrated but emitted no action tags — nudge it to act
                        currentUserPrompt = "[fresh screenshot attached. you described what you'd do but emitted no action tags. take concrete action now using APPLESCRIPT, CLICK, TYPE, or other available tags. if the task is somehow already complete, emit [TASK_DONE] with a spoken confirmation. original task: \(transcript).]"
                        continue
                    } else {
                        // Actions ran but no [TASK_DONE] yet — auto-screenshot and continue
                        currentUserPrompt = "[fresh screenshot attached. \(executedActionSummary). continue working on the task. when the task is fully complete, emit [TASK_DONE] followed by a spoken summary of what was done. original task: \(transcript).]"
                        continue
                    }
                }

                if iterationCount >= maxIterations {
                    let loopMsg = "i think i'm going in circles, stopping here"
                    try? await geminiTTSClient.speakText(loopMsg)
                }

                // History was saved eagerly per-iteration (after each TTS).
                // Just track analytics from the consolidated final state.
                let combinedResponse = allSpokenText.joined(separator: " ")
                MickyAnalytics.trackAIResponseReceived(response: combinedResponse)

            } catch is CancellationError {
                // User spoke again — interrupted
            } catch {
                MickyAnalytics.trackResponseError(error: error.localizedDescription)
                print("⚠️ Companion response error: \(error)")
                speakCreditsErrorFallback()
            }

            if !Task.isCancelled {
                voiceState = .idle
                scheduleTransientHideIfNeeded()
            }
        }
    }

    /// Converts a screenshot pixel coordinate to AppKit global coords and
    /// sets the pointing animation target.
    private func applyPointingCoordinates(x: Int, y: Int, screenNumber: Int?, screenCaptures: [CompanionScreenCapture], label: String?) {
        let targetScreenCapture: CompanionScreenCapture? = {
            if let screenNumber, screenNumber >= 1, screenNumber <= screenCaptures.count {
                return screenCaptures[screenNumber - 1]
            }
            return screenCaptures.first(where: { $0.isCursorScreen })
        }()

        guard let capture = targetScreenCapture else { return }

        let screenshotWidth  = CGFloat(capture.screenshotWidthInPixels)
        let screenshotHeight = CGFloat(capture.screenshotHeightInPixels)
        let displayWidth     = CGFloat(capture.displayWidthInPoints)
        let displayHeight    = CGFloat(capture.displayHeightInPoints)
        let displayFrame     = capture.displayFrame

        let clampedX = max(0, min(CGFloat(x), screenshotWidth))
        let clampedY = max(0, min(CGFloat(y), screenshotHeight))
        let displayLocalX = clampedX * (displayWidth  / screenshotWidth)
        let displayLocalY = clampedY * (displayHeight / screenshotHeight)
        let appKitY = displayHeight - displayLocalY

        detectedElementScreenLocation = CGPoint(x: displayLocalX + displayFrame.origin.x, y: appKitY + displayFrame.origin.y)
        detectedElementDisplayFrame = displayFrame
        MickyAnalytics.trackElementPointed(elementLabel: label)
    }

    /// If the cursor is in transient mode (user toggled "Show Micky" off),
    /// waits for TTS playback and any pointing animation to finish, then
    /// fades out the overlay after a 1-second pause. Cancelled automatically
    /// if the user starts another push-to-talk interaction.
    private func scheduleTransientHideIfNeeded() {
        guard !isMickyCursorEnabled && isOverlayVisible else { return }

        transientHideTask?.cancel()
        transientHideTask = Task {
            // Wait for TTS audio to finish playing
            while geminiTTSClient.isPlaying {
                try? await Task.sleep(nanoseconds: 200_000_000)
                guard !Task.isCancelled else { return }
            }

            // Wait for pointing animation to finish (location is cleared
            // when the buddy flies back to the cursor)
            while detectedElementScreenLocation != nil {
                try? await Task.sleep(nanoseconds: 200_000_000)
                guard !Task.isCancelled else { return }
            }

            // Pause 1s after everything finishes, then fade out
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            guard !Task.isCancelled else { return }
            overlayWindowManager.fadeOutAndHideOverlay()
            isOverlayVisible = false
        }
    }

    // MARK: - Context Compaction

    /// Compresses old conversation turns when history grows large, keeping
    /// the last 3 turns verbatim and collapsing earlier ones into a summary
    /// block. Prevents context-window overflow on long autonomous tasks.
    private func compactConversationHistoryIfNeeded() {
        let totalCharacterCount = conversationHistory.reduce(0) {
            $0 + $1.userTranscript.count + $1.assistantResponse.count
        }
        let shouldCompact = conversationHistory.count > 7 || totalCharacterCount > 25_000
        guard shouldCompact, conversationHistory.count > 4 else { return }

        let turnsToCompress = Array(conversationHistory.dropLast(3))
        let compressedContent = turnsToCompress.enumerated().map { index, entry in
            "Turn \(index + 1): user said \"\(entry.userTranscript.prefix(120))\". " +
            "micky responded: \"\(entry.assistantResponse.prefix(250))\""
        }.joined(separator: "\n")

        let compressedEntry = (
            userTranscript: "[Earlier conversation compressed — \(turnsToCompress.count) turns]",
            assistantResponse: compressedContent
        )
        conversationHistory = [compressedEntry] + Array(conversationHistory.suffix(3))
        print("🗜️ Compacted \(turnsToCompress.count) old history entries (\(totalCharacterCount) chars → summary)")
    }

    // MARK: - Memory Helpers

    /// Extracts meaningful keywords from a transcript for memory indexing.
    /// Filters stop words and short tokens; returns up to 8 unique words.
    private func extractKeywordsForMemory(from text: String) -> [String] {
        let stopWords = Set([
            "the", "and", "for", "with", "this", "that", "have", "from",
            "they", "will", "been", "were", "are", "was", "you", "can",
            "not", "but", "its", "also", "then", "into", "open", "just"
        ])
        var seen: Set<String> = []
        return text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= 4 && !stopWords.contains($0) }
            .compactMap { word -> String? in
                guard seen.insert(word).inserted else { return nil }
                return word
            }
            .prefix(8)
            .map { $0 }
    }

    /// Speaks a hardcoded error message using macOS system TTS when the
    /// Gemini API is unavailable. Uses NSSpeechSynthesizer as a fallback.
    private func speakCreditsErrorFallback() {
        let utterance = "I'm all out of credits. Please DM Varun and tell him to bring me back to life."
        let synthesizer = NSSpeechSynthesizer()
        synthesizer.startSpeaking(utterance)
        voiceState = .responding
    }

    // MARK: - Onboarding Video

    /// Sets up the onboarding video player, starts playback, and schedules
    /// the demo interaction at 40s. Called by BlueCursorView when onboarding starts.
    func setupOnboardingVideo() {
        // Skip video entirely — go straight to the "press ctrl+option" prompt
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            self.startOnboardingPromptStream()
        }
    }

    func tearDownOnboardingVideo() {
        showOnboardingVideo = false
        onboardingVideoOpacity = 0.0
    }

    private func startOnboardingPromptStream() {
        let message = "press control + option and introduce yourself"
        onboardingPromptText = ""
        showOnboardingPrompt = true
        onboardingPromptOpacity = 0.0

        withAnimation(.easeIn(duration: 0.4)) {
            onboardingPromptOpacity = 1.0
        }

        var currentIndex = 0
        Timer.scheduledTimer(withTimeInterval: 0.03, repeats: true) { timer in
            guard currentIndex < message.count else {
                timer.invalidate()
                // Auto-dismiss after 10 seconds
                DispatchQueue.main.asyncAfter(deadline: .now() + 10.0) {
                    guard self.showOnboardingPrompt else { return }
                    withAnimation(.easeOut(duration: 0.3)) {
                        self.onboardingPromptOpacity = 0.0
                    }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                        self.showOnboardingPrompt = false
                        self.onboardingPromptText = ""
                    }
                }
                return
            }
            let index = message.index(message.startIndex, offsetBy: currentIndex)
            self.onboardingPromptText.append(message[index])
            currentIndex += 1
        }
    }

    /// Gradually raises an AVPlayer's volume from its current level to the
    /// target over the specified duration, creating a smooth audio fade-in.
    private func fadeInVideoAudio(player: AVPlayer, targetVolume: Float, duration: Double) {
        let steps = 20
        let stepInterval = duration / Double(steps)
        let volumeIncrement = (targetVolume - player.volume) / Float(steps)
        var stepsRemaining = steps

        Timer.scheduledTimer(withTimeInterval: stepInterval, repeats: true) { timer in
            stepsRemaining -= 1
            player.volume += volumeIncrement

            if stepsRemaining <= 0 {
                timer.invalidate()
                player.volume = targetVolume
            }
        }
    }

    // MARK: - Onboarding Demo Interaction

    private static let onboardingDemoSystemPrompt = """
    you're micky, a small blue cursor buddy living on the user's screen. you're showing off during onboarding — look at their screen and find ONE specific, concrete thing to point at. pick something with a clear name or identity: a specific app icon (say its name), a specific word or phrase of text you can read, a specific filename, a specific button label, a specific tab title, a specific image you can describe. do NOT point at vague things like "a window" or "some text" — be specific about exactly what you see.

    make a short quirky 3-6 word observation about the specific thing you picked — something fun, playful, or curious that shows you actually read/recognized it. no emojis ever. NEVER quote or repeat text you see on screen — just react to it. keep it to 6 words max, no exceptions.

    CRITICAL COORDINATE RULE: you MUST only pick elements near the CENTER of the screen. your x coordinate must be between 20%-80% of the image width. your y coordinate must be between 20%-80% of the image height. do NOT pick anything in the top 20%, bottom 20%, left 20%, or right 20% of the screen. no menu bar items, no dock icons, no sidebar items, no items near any edge. only things clearly in the middle area of the screen. if the only interesting things are near the edges, pick something boring in the center instead.

    respond with ONLY your short comment followed by the coordinate tag. nothing else. all lowercase.

    format: your comment [POINT:x,y:label]

    the screenshot images are labeled with their pixel dimensions. use those dimensions as the coordinate space. origin (0,0) is top-left. x increases rightward, y increases downward.
    """

    /// Captures a screenshot and asks Claude to find something interesting to
    /// point at, then triggers the buddy's flight animation. Used during
    /// onboarding to demo the pointing feature while the intro video plays.
    func performOnboardingDemoInteraction() {
        // Don't interrupt an active voice response
        guard voiceState == .idle || voiceState == .responding else { return }

        Task {
            do {
                let screenCaptures = try await CompanionScreenCaptureUtility.captureAllScreensAsJPEG()

                // Only send the cursor screen so Claude can't pick something
                // on a different monitor that we can't point at.
                guard let cursorScreenCapture = screenCaptures.first(where: { $0.isCursorScreen }) else {
                    print("🎯 Onboarding demo: no cursor screen found")
                    return
                }

                let dimensionInfo = " (image dimensions: \(cursorScreenCapture.screenshotWidthInPixels)x\(cursorScreenCapture.screenshotHeightInPixels) pixels)"
                let labeledImages = [(data: cursorScreenCapture.imageData, label: cursorScreenCapture.label + dimensionInfo)]

                let (fullResponseText, _) = try await geminiAPI.analyzeImageStreaming(
                    images: labeledImages,
                    systemPrompt: Self.onboardingDemoSystemPrompt,
                    userPrompt: "look around my screen and find something interesting to point at",
                    onTextChunk: { _ in }
                )

                let parsed = AgenticTagParser.parse(fullResponseText)

                guard let pointTag = parsed.pointTag,
                      case .point(let px, let py, let label, _) = pointTag,
                      (px > 0 || py > 0) else {
                    print("🎯 Onboarding demo: no element to point at")
                    return
                }

                let screenshotWidth = CGFloat(cursorScreenCapture.screenshotWidthInPixels)
                let screenshotHeight = CGFloat(cursorScreenCapture.screenshotHeightInPixels)
                let displayWidth = CGFloat(cursorScreenCapture.displayWidthInPoints)
                let displayHeight = CGFloat(cursorScreenCapture.displayHeightInPoints)
                let displayFrame = cursorScreenCapture.displayFrame

                let clampedX = max(0, min(CGFloat(px), screenshotWidth))
                let clampedY = max(0, min(CGFloat(py), screenshotHeight))
                let displayLocalX = clampedX * (displayWidth / screenshotWidth)
                let displayLocalY = clampedY * (displayHeight / screenshotHeight)
                let appKitY = displayHeight - displayLocalY
                let globalLocation = CGPoint(
                    x: displayLocalX + displayFrame.origin.x,
                    y: appKitY + displayFrame.origin.y
                )

                detectedElementBubbleText = parsed.spokenText
                detectedElementScreenLocation = globalLocation
                detectedElementDisplayFrame = displayFrame
                print("🎯 Onboarding demo: pointing at \"\(label ?? "element")\" — \"\(parsed.spokenText)\"")
            } catch {
                print("⚠️ Onboarding demo error: \(error)")
            }
        }
    }
}
