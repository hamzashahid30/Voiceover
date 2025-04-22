import streamlit as st
import torchaudio
from gtts import gTTS
from tortoise.api import TextToSpeech
import time

# --------------------------
# App Title & Config
# --------------------------
st.set_page_config(page_title="AI Voice Generator (English)", layout="wide")
st.title("🎤 AI English VoiceOver & Cloner")
st.write("Generate TTS or Clone Your Voice - 100% Free")

# --------------------------
# Core Functions
# --------------------------
def text_to_speech(text):
    """Convert text to speech using Google TTS"""
    tts = gTTS(text=text, lang='en', slow=False)
    output_file = f"tts_output_{int(time.time())}.mp3"
    tts.save(output_file)
    return output_file

def clone_voice(text, voice_file):
    """Clone voice using Tortoise TTS"""
    # Save uploaded voice
    with open("user_voice.wav", "wb") as f:
        f.write(voice_file.read())
    
    # Initialize TTS model
    tts = TextToSpeech()
    
    # Extract voice features
    voice_samples, conditioning_latents = tts.get_conditioning_latents("user_voice.wav")
    
    # Generate cloned voice
    gen = tts.tts_with_preset(text, 
                             voice_samples=voice_samples,
                             conditioning_latents=conditioning_latents,
                             preset='fast')
    
    # Save output
    output_file = f"clone_output_{int(time.time())}.wav"
    torchaudio.save(output_file, gen.squeeze(0).cpu(), 24000)
    return output_file

# --------------------------
# App Interface
# --------------------------
tab1, tab2 = st.tabs(["Text-to-Speech", "Voice Cloning"])

# TAB 1: Text-to-Speech (TTS)
with tab1:
    st.header("🔊 Text-to-Speech (English)")
    tts_text = st.text_area("Enter your text here...", height=150)
    
    if st.button("Generate VoiceOver", key="tts_btn"):
        if tts_text:
            with st.spinner("Generating audio..."):
                output_path = text_to_speech(tts_text)
                st.audio(output_path, format="audio/mp3")
                st.success("✅ Done! Download below")
        else:
            st.warning("Please enter some text first!")

# TAB 2: Voice Cloning
with tab2:
    st.header("🎤 Voice Cloning")
    st.info("Upload a 5-10 sec clear voice sample (.wav)")
    
    clone_text = st.text_area("Text to clone in your voice...", height=150)
    voice_sample = st.file_uploader("Upload your voice (.wav)", type=["wav"])
    
    if st.button("Clone My Voice", key="clone_btn"):
        if clone_text and voice_sample:
            with st.spinner("Cloning your voice (may take 1-2 mins)..."):
                output_path = clone_voice(clone_text, voice_sample)
                st.audio(output_path, format="audio/wav")
                st.success("✅ Cloning Complete! Listen above")
        else:
            st.error("Please provide both text and voice sample!")

# --------------------------
# How to Use Guide
# --------------------------
st.markdown("---")
st.subheader("📌 Instructions")
st.write("""
1. **For Text-to-Speech**: Just type text → Click "Generate VoiceOver"  
2. **For Voice Cloning**:  
   - Record a **5-10 sec clear English** voice sample (.wav)  
   - Upload it + Enter text → Click "Clone My Voice"  
3. **Note**: Cloning is slower (~1-2 mins) due to AI processing  
""")
