//
//  AgenticCoordinateMapper.swift
//  leanring-buddy
//
//  Converts screenshot pixel coordinates to CGEvent global coordinates.
//  CGEvent uses top-left origin; AppKit uses bottom-left — these are NOT
//  interchangeable. This file handles only the CGEvent path for real clicks.
//

import AppKit
import CoreGraphics

enum AgenticCoordinateMapper {

    /// Converts a screenshot pixel coordinate to CGEvent global coordinates.
    /// Requires the actual screenshot pixel dimensions so the scale is exact —
    /// never guess based on backingScaleFactor alone.
    ///
    /// - Parameters:
    ///   - screenshotX: X coordinate in screenshot pixel space (0 = left edge)
    ///   - screenshotY: Y coordinate in screenshot pixel space (0 = top edge)
    ///   - actualScreenshotWidth:  real pixel width of the screenshot image
    ///   - actualScreenshotHeight: real pixel height of the screenshot image
    ///   - screenIndex: 1-based screen index, or nil for cursor screen
    static func cgEventGlobal(
        screenshotX: Int,
        screenshotY: Int,
        actualScreenshotWidth: Int,
        actualScreenshotHeight: Int,
        screenIndex: Int?
    ) -> CGPoint {
        let screens = NSScreen.screens

        let targetScreen: NSScreen
        if let index = screenIndex, index >= 1, index <= screens.count {
            targetScreen = screens[index - 1]
        } else {
            let mouseLocation = NSEvent.mouseLocation
            targetScreen = screens.first(where: { $0.frame.contains(mouseLocation) }) ?? screens[0]
        }

        let displayID = targetScreen.displayID
        let cgDisplayFrame = CGDisplayBounds(displayID)

        let displayWidthPoints  = targetScreen.frame.width
        let displayHeightPoints = targetScreen.frame.height

        // Use real screenshot dimensions — not a guess based on scale factor
        let scaleX = displayWidthPoints  / CGFloat(actualScreenshotWidth)
        let scaleY = displayHeightPoints / CGFloat(actualScreenshotHeight)

        let localX = CGFloat(screenshotX) * scaleX
        let localY = CGFloat(screenshotY) * scaleY

        print("📐 CoordMapper: screenshot=\(actualScreenshotWidth)x\(actualScreenshotHeight)px display=\(Int(displayWidthPoints))x\(Int(displayHeightPoints))pts scale=(\(String(format:"%.2f",scaleX)),\(String(format:"%.2f",scaleY))) → cgEvent(\(Int(cgDisplayFrame.origin.x + localX)),\(Int(cgDisplayFrame.origin.y + localY)))")

        return CGPoint(
            x: cgDisplayFrame.origin.x + localX,
            y: cgDisplayFrame.origin.y + localY
        )
    }
}
