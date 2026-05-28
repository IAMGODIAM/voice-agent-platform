os.makedirs(r"C:\Users\User\Desktop\israel-voice-workspace\sesame-finetune\data", exist_ok=True)
env_content = """CSM_REPO_PATH=C:/Users/User/Desktop/israel-voice-workspace/csm
MIMI_SAMPLE_RATE=24000
BACKBONE_FLAVOR=llama-1B
DECODER_FLAVOR=llama-100M
TEXT_VOCAB_SIZE=128256
AUDIO_VOCAB_SIZE=2051
AUDIO_NUM_CODEBOOKS=32
"""
with open(r"C:\Users\User\Desktop\israel-voice-workspace\sesame-finetune\.env", "w") as f:
    f.write(env_content)
print("Created .env")
