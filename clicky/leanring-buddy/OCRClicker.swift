//
//  OCRClicker.swift
//  leanring-buddy
//
//  Click a UI element by the visible text on screen, using macOS Vision
//  framework OCR. This is the fallback for AXCLICK when the accessibility
//  tree doesn't surface the element (Electron apps with thin AX trees,
//  custom-rendered chat lists, etc.).
//
//  The pipeline never trusts a vision model's pixel-coordinate guesses:
//  Vision OCR returns the *actual* bounding box of the text it recognizes,
//  so a click at the box's center lands on the visible text every time —
//  the exact technique the original farzaa/clicky uses to get the cursor
//  to the right place, but adapted to run locally instead of through
//  Anthropic's Computer Use API.
//
//  Vision returns normalized coordinates with a bottom-left origin; we
//  convert them to top-left screenshot pixel space, then use the existing
//  AgenticCoordinateMapper to map screenshot pixels → display points so
//  clicks land in the correct screen location regardless of HiDPI scale.
//

import AppKit
import CoreGraphics
import Foundation
import Vision

@MainActor
enum OCRClicker {

    enum Result {
        /// Found the text and clicked at its bounding-box center.
        case clicked(matchedText: String, displayPoint: CGPoint)
        /// OCR ran but no recognized text contained the search string.
        case noMatch(visibleTextSample: [String])
        /// OCR couldn't run at all (bad image data, framework failure, no screenshot supplied).
        case failed(reason: String)
    }

    /// Search for `searchText` in the supplied screenshot and click the
    /// center of the matching text's bounding box.
    ///
    /// - Parameters:
    ///   - searchText: substring to find (case-insensitive, whitespace-trimmed).
    ///   - screenshotData: raw JPEG/PNG bytes of the most recent screenshot.
    ///   - screenshotWidth/Height: the actual pixel dimensions of that screenshot.
    ///   - screenIndex: 1-based screen index to map clicks to, or nil for the cursor screen.
    static func click(
        searchText rawSearchText: String,
        screenshotData: Data,
        screenshotWidth: Int,
        screenshotHeight: Int,
        screenIndex: Int? = nil
    ) -> Result {
        let normalizedSearch = rawSearchText
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard !normalizedSearch.isEmpty else {
            return .failed(reason: "empty search text")
        }

        guard let cgImage = decodeCGImage(from: screenshotData) else {
            return .failed(reason: "couldn't decode screenshot bytes")
        }

        let textRecognitionRequest = VNRecognizeTextRequest()
        textRecognitionRequest.recognitionLevel = .accurate
        textRecognitionRequest.usesLanguageCorrection = false
        textRecognitionRequest.recognitionLanguages = ["en-US"]

        let imageRequestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try imageRequestHandler.perform([textRecognitionRequest])
        } catch {
            return .failed(reason: "Vision OCR failed: \(error.localizedDescription)")
        }

        guard let textObservations = textRecognitionRequest.results else {
            return .failed(reason: "Vision returned no observation set")
        }

        // Collect every recognized string for the no-match diagnostic, AND
        // pick the best match. We prefer:
        //   1. exact-match-of-substring with highest OCR confidence
        //   2. shorter recognized text (a row labeled "Canara NSUT" beats a
        //      paragraph containing "Canara NSUT" as part of a longer phrase)
        var allRecognizedTexts: [String] = []
        var candidateMatches: [(text: String, observation: VNRecognizedTextObservation, confidence: Float)] = []

        for textObservation in textObservations {
            guard let topRecognizedText = textObservation.topCandidates(1).first else { continue }
            let recognizedText = topRecognizedText.string
            allRecognizedTexts.append(recognizedText)

            if recognizedText.lowercased().contains(normalizedSearch) {
                candidateMatches.append((
                    text: recognizedText,
                    observation: textObservation,
                    confidence: topRecognizedText.confidence
                ))
            }
        }

        guard !candidateMatches.isEmpty else {
            // De-duplicate, sort, and trim the visible-text sample for the report
            let dedupedSample = Array(Set(allRecognizedTexts))
                .filter { !$0.isEmpty && $0.count <= 80 }
                .sorted()
                .prefix(40)
            return .noMatch(visibleTextSample: Array(dedupedSample))
        }

        // Score: confidence (0-1) - length penalty (longer = lower score)
        let bestMatch = candidateMatches.max(by: { lhs, rhs in
            let lhsScore = lhs.confidence - Float(lhs.text.count) * 0.005
            let rhsScore = rhs.confidence - Float(rhs.text.count) * 0.005
            return lhsScore < rhsScore
        })!

        // Vision uses normalized [0,1] coords with bottom-left origin.
        // Convert to top-left screenshot pixel coords for AgenticCoordinateMapper.
        let normalizedBoundingBox = bestMatch.observation.boundingBox
        let centerXNormalized = normalizedBoundingBox.midX
        let centerYNormalizedBottomLeft = normalizedBoundingBox.midY

        let screenshotPixelX = Int(centerXNormalized * CGFloat(screenshotWidth))
        let screenshotPixelYTopLeft = Int((1.0 - centerYNormalizedBottomLeft) * CGFloat(screenshotHeight))

        let displayPoint = AgenticCoordinateMapper.cgEventGlobal(
            screenshotX: screenshotPixelX,
            screenshotY: screenshotPixelYTopLeft,
            actualScreenshotWidth: screenshotWidth,
            actualScreenshotHeight: screenshotHeight,
            screenIndex: screenIndex
        )

        postCGEventClick(at: displayPoint)
        return .clicked(matchedText: bestMatch.text, displayPoint: displayPoint)
    }

    // MARK: - Helpers

    private static func decodeCGImage(from imageData: Data) -> CGImage? {
        guard let bitmapImageRep = NSBitmapImageRep(data: imageData) else { return nil }
        return bitmapImageRep.cgImage
    }

    private static func postCGEventClick(at point: CGPoint) {
        let source = CGEventSource(stateID: .hidSystemState)
        let down = CGEvent(mouseEventSource: source, mouseType: .leftMouseDown, mouseCursorPosition: point, mouseButton: .left)
        let up   = CGEvent(mouseEventSource: source, mouseType: .leftMouseUp,   mouseCursorPosition: point, mouseButton: .left)
        down?.post(tap: .cghidEventTap)
        up?.post(tap: .cghidEventTap)
    }
}
