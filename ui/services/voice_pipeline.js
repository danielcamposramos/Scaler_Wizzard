/*
 * VoicePipeline provides queued playback for cockpit alerts.
 * It relies on the Web Speech API when available and gracefully
 * degrades to text callbacks when synthesis is not supported.
 */

export class VoicePipeline {
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

  enqueue({ message, language = 'en-US', voiceName }) {
    this.voiceQueue.push({ message, language, voiceName });
    this._processQueue();
  }

  clear() {
    this.voiceQueue = [];
    if (this.synth && this.synth.speaking) {
      this.synth.cancel();
    }
    this.isSpeaking = false;
  }

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

  _onComplete() {
    this.isSpeaking = false;
    if (this.voiceQueue.length > 0) {
      setTimeout(() => this._processQueue(), 50);
    }
  }

  _resolveLanguage(requested) {
    if (this.supportedLanguages.includes(requested)) {
      return requested;
    }
    const [preferred] = this.supportedLanguages;
    return preferred || 'en-US';
  }

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
