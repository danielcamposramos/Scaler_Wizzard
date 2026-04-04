/**
 * Scaler Wizard: Throttled Voice Pipeline
 * Manages audio alerts with cognitive-load management.
 */

class VoicePipeline {
  constructor() {
    this.synth = window.speechSynthesis;
    this.persona = 'Calm Architect';
    this.supportedLanguages = ['en-US', 'pt-BR', 'es-ES', 'fr-FR', 'de-DE', 'zh-CN'];
    this.currentLanguage = 'en-US';
    this.isSpeaking = false;
    this.voiceQueue = [];
    this.maxQueueLength = 3;
  }

  narrate(telegram) {
    // Check if voice is the active prioritized modality
    if (this.isSpeaking) {
      if (this.voiceQueue.length < this.maxQueueLength) {
        this.voiceQueue.push(telegram);
      }
      return false;
    }

    const text = telegram.qualityMetrics.canonicalSentence;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = this.currentLanguage;
    
    // Vibe-adaptive emotional cues
    utterance.rate = telegram.status === 'warning' ? 1.1 : 0.95;
    utterance.pitch = telegram.status === 'stopped' ? 0.8 : 1.0;

    const voices = this.synth.getVoices();
    utterance.voice = voices.find(v => v.lang === this.currentLanguage) || voices[0];

    this.isSpeaking = true;
    this.synth.speak(utterance);

    utterance.onend = () => {
      this.isSpeaking = false;
      this.processQueue();
    };

    utterance.onerror = (event) => {
      console.error('Speech synthesis error:', event);
      this.isSpeaking = false;
      this.processQueue();
    };

    return true;
  }

  processQueue() {
    if (this.voiceQueue.length > 0 && !this.isSpeaking) {
      this.narrate(this.voiceQueue.shift());
    }
  }
}

export default VoicePipeline;