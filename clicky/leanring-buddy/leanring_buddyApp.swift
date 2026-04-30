//
//  leanring_buddyApp.swift
//  leanring-buddy
//
//  Menu bar-only companion app. No dock icon, no main window — just an
//  always-available status item in the macOS menu bar. Clicking the icon
//  opens a floating panel with companion voice controls.
//

import ServiceManagement
import SwiftUI
import Sparkle

@main
struct leanring_buddyApp: App {
    @NSApplicationDelegateAdaptor(CompanionAppDelegate.self) var appDelegate

    var body: some Scene {
        // The app lives entirely in the menu bar panel managed by the AppDelegate.
        // This empty Settings scene satisfies SwiftUI's requirement for at least
        // one scene but is never shown (LSUIElement=true removes the app menu).
        Settings {
            EmptyView()
        }
    }
}

/// Manages the companion lifecycle: creates the menu bar panel and starts
/// the companion voice pipeline on launch.
@MainActor
final class CompanionAppDelegate: NSObject, NSApplicationDelegate {
    private var menuBarPanelManager: MenuBarPanelManager?
    private let companionManager = CompanionManager()
    private var sparkleUpdaterController: SPUStandardUpdaterController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        print("🎯 Micky: Starting...")
        print("🎯 Micky: Version \(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "unknown")")

        UserDefaults.standard.register(defaults: ["NSInitialToolTipDelay": 0])

        triggerFileSystemPermissionPromptsIfNeeded()

        MickyAnalytics.configure()
        MickyAnalytics.trackAppOpened()

        menuBarPanelManager = MenuBarPanelManager(companionManager: companionManager)
        companionManager.start()
        // Auto-open the panel if the user still needs to do something:
        // either they haven't onboarded yet, or permissions were revoked.
        if !companionManager.hasCompletedOnboarding || !companionManager.allPermissionsGranted {
            menuBarPanelManager?.showPanelOnLaunch()
        }
        registerAsLoginItemIfNeeded()
        // startSparkleUpdater()
    }

    func applicationWillTerminate(_ notification: Notification) {
        companionManager.stop()
    }

    /// Reads the user-protected directories Clicky operates over so macOS
    /// surfaces its TCC consent prompts immediately on first launch instead of
    /// failing silently later when the agent tries to `open` a path inside
    /// them. Without this, an unsandboxed app trying to open `~/Downloads/...`
    /// via LaunchServices gets `error -54` because TCC never asked the user
    /// for consent. After the user clicks Allow once per directory, every
    /// subsequent open works for the lifetime of the bundle ID.
    private func triggerFileSystemPermissionPromptsIfNeeded() {
        let realUserHome = NSHomeDirectoryForUser(NSUserName()) ?? NSHomeDirectory()
        let userProtectedDirectories = [
            "\(realUserHome)/Downloads",
            "\(realUserHome)/Desktop",
            "\(realUserHome)/Documents",
        ]
        for directoryPath in userProtectedDirectories {
            // The act of reading the contents is what trips the TCC prompt.
            // We discard the result; success or failure both leave TCC in the
            // right state for subsequent calls.
            _ = try? FileManager.default.contentsOfDirectory(atPath: directoryPath)
        }
    }

    /// Registers the app as a login item so it launches automatically on
    /// startup. Uses SMAppService which shows the app in System Settings >
    /// General > Login Items, letting the user toggle it off if they want.
    private func registerAsLoginItemIfNeeded() {
        let loginItemService = SMAppService.mainApp
        if loginItemService.status != .enabled {
            do {
                try loginItemService.register()
                print("🎯 Micky: Registered as login item")
            } catch {
                print("⚠️ Micky: Failed to register as login item: \(error)")
            }
        }
    }

    private func startSparkleUpdater() {
        let updaterController = SPUStandardUpdaterController(
            startingUpdater: false,
            updaterDelegate: nil,
            userDriverDelegate: nil
        )
        self.sparkleUpdaterController = updaterController

        do {
            try updaterController.updater.start()
        } catch {
            print("⚠️ Micky: Sparkle updater failed to start: \(error)")
        }
    }
}
