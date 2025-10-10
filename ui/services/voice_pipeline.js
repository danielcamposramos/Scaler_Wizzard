/**
 * VoicePipeline provides queued playback for cockpit alerts.
 * It relies on the Web Speech API when available and gracefully
 * degrades to text callbacks when synthesis is not supported.
 * @class
 */
export class VoicePipeline {
  /**
   * Creates an instance of VoicePipeline.
   * @param {object} [options={}] - Configuration options for the pipeline.
   * @param {string[]} [options.supportedLanguages=['en-US', 'pt-BR', ...]] - A list of supported language codes.
   * @param {function} [options.onFallback] - A callback function to execute when speech synthesis is not available.
   */
  constructor(options = {}) {
    this.supportedLanguages = options.supportedLanguages || [
      'en-US',
      'pt-BR',
      'es-ES',
      'fr-FR',
      'de-DE',
      'zh-CN'
    ];
    this.voiceQueue = [];
    this.isSpeaking = false;
    this.onFallback = options.onFallback || (() => {});
    this.synth = typeof window !== 'undefined' ? window.speechSynthesis : null;
  }

  /**
   * Adds a message to the playback queue.
   * @param {object} options - The message options.
   * @param {string} options.message - The text message to be spoken.
   * @param {string} [options.language='en-US'] - The desired language for the message.
   * @param {string} [options.voiceName] - The specific voice name to use.
   */
  enqueue({ message, language = 'en-US', voiceName }) {
    this.voiceQueue.push({ message, language, voiceName });
    this._processQueue();
  }

  /**
   * Clears the message queue and stops any currently speaking utterance.
   */
  clear() {
    this.voiceQueue = [];
    if (this.synth && this.synth.speaking) {
      this.synth.cancel();
    }
    this.isSpeaking = false;
  }

  /**
   * Processes the next message in the queue.
   * If speech synthesis is not available, it calls the fallback function.
   * @private
   */
  _processQueue() {
    if (this.isSpeaking || this.voiceQueue.length === 0) {
      return;
    }

    const next = this.voiceQueue.shift();

    if (!this.synth || typeof SpeechSynthesisUtterance === 'undefined') {
      this.onFallback(next);
      this._onComplete();
      return;
    }

    const utterance = new SpeechSynthesisUtterance(next.message);
    utterance.lang = this._resolveLanguage(next.language);
    utterance.voice = this._resolveVoice(utterance.lang, next.voiceName);
    utterance.onend = () => this._onComplete();
    utterance.onerror = () => {
      this.onFallback(next);
      this._onComplete();
    };

    this.isSpeaking = true;
    this.synth.speak(utterance);
  }

  /**
   * Handles the completion of a speech utterance.
   * It resets the speaking state and attempts to process the next item in the queue.
   * @private
   */
  _onComplete() {
    this.isSpeaking = false;
    if (this.voiceQueue.length > 0) {
      setTimeout(() => this._processQueue(), 50);
    }
  }

  /**
   * Resolves the language to use for an utterance.
   * Falls back to the first supported language if the requested one is not available.
   * @param {string} requested - The requested language code.
   * @returns {string} The resolved language code.
   * @private
   */
  _resolveLanguage(requested) {
    if (this.supportedLanguages.includes(requested)) {
      return requested;
    }
    const [preferred] = this.supportedLanguages;
    return preferred || 'en-US';
  }

  /**
   * Finds a suitable voice for the given language and optional voice name.
   * @param {string} language - The language for the voice.
   * @param {string} [voiceName] - The preferred name of the voice.
   * @returns {SpeechSynthesisVoice|null} The found voice, or null if none are available.
   * @private
   */
  _resolveVoice(language, voiceName) {
    if (!this.synth) {
      return null;
    }
    const voices = this.synth.getVoices();
    if (!voices || voices.length === 0) {
      return null;
    }

    if (voiceName) {
      const byName = voices.find((voice) => voice.name === voiceName);
      if (byName) {
        return byName;
      }
    }

    const byLang = voices.find((voice) => voice.lang === language);
    return byLang || voices[0];
  }
}

export default VoicePipeline;
