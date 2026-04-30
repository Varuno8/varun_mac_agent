//
//  GeminiTTSClient.swift
//  leanring-buddy
//
//  Text-to-speech using Gemini's native audio output modality.
//  Replaces ElevenLabsTTSClient.swift.
//
//  The Cloudflare Worker proxy handles Vertex AI auth and forwards the
//  request. The response is raw PCM audio wrapped in a WAV container
//  before playback via AVAudioPlayer.
//

import AVFoundation
import Foundation

@MainActor
final class GeminiTTSClient {
    private let proxyURL: URL
    private let session: URLSession
    private let voiceName: String

    /// The audio player for the current TTS playback.
    private var audioPlayer: AVAudioPlayer?

    init(proxyURL: String, voiceName: String = "Kore") {
        self.proxyURL = URL(string: proxyURL)!
        self.voiceName = voiceName

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        self.session = URLSession(configuration: config)
    }

    /// Sends `text` to Gemini TTS and plays the resulting audio.
    func speakText(_ text: String) async throws {
        var request = URLRequest(url: proxyURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "text": text,
            "voiceName": voiceName
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw NSError(domain: "GeminiTTS", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "Invalid response"])
        }
        guard (200...299).contains(httpResponse.statusCode) else {
            let errorBody = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw NSError(domain: "GeminiTTS", code: httpResponse.statusCode,
                          userInfo: [NSLocalizedDescriptionKey: "TTS API error (\(httpResponse.statusCode)): \(errorBody)"])
        }

        try Task.checkCancellation()

        // The Worker returns a WAV file directly. Play it via AVAudioPlayer.
        let player = try AVAudioPlayer(data: data)
        self.audioPlayer = player
        player.play()
        print("🔊 Gemini TTS: playing \(data.count / 1024)KB audio")
    }

    /// Whether TTS audio is currently playing.
    var isPlaying: Bool {
        audioPlayer?.isPlaying ?? false
    }

    /// Stops any in-progress playback immediately.
    func stopPlayback() {
        audioPlayer?.stop()
        audioPlayer = nil
    }
}
