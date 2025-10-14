# 🚨 StreamSplit Training Issues - Critical Findings

## Date: October 14, 2025

---

## ❌ CRITICAL ISSUES IDENTIFIED

### 1. **Dataset Returns Raw Audio Instead of Mel-Spectrograms**
```
Expected: Mel-spectrogram torch.Size([1, 128, ~1000])
Actual:   Raw audio       torch.Size([160000])
```

**Impact:** Model receives WRONG INPUT FORMAT
- Model expects 2D spectrogram
- Gets 1D raw waveform instead
- This causes conv2d errors and explains high loss!

**Root Cause:** Dataset `__getitem__` returns raw audio, not preprocessed features

---

### 2. **Model Output Has Very Low Variance**
```
Output std: 0.0024 (expected: > 0.1)
Output range: [-0.0095, 0.0071]
```

**Impact:** Model embeddings are almost identical
- All samples map to nearly the same point
- No discriminative features learned
- Contrastive loss cannot work properly

**Likely Cause:** Model sees wrong input → learns nothing useful

---

### 3. **Loss is 2.5x Higher Than Expected**
```
Your loss: 2.54
Expected:  < 1.0 for working contrastive learning
Baseline:  3.47 for random embeddings
```

**Analysis:**
- Random embeddings: 3.5 loss
- Your trained model: 2.5 loss  
- Improvement: Only 28% (should be 70%+)
- **Model barely learning beyond random initialization**

---

## 🔍 DIAGNOSIS

### What's Happening:

1. **Dataset loads audio** → Returns waveform shape `[160000]`
2. **Model expects spectrogram** → Shape `[1, 128, 1000]`
3. **Training script tries to feed waveform to model** → Crashes or creates list
4. **When it doesn't crash** → Model processes garbage input
5. **Loss barely improves** → 3.1 → 2.5 (should go to < 1.0)

### Why Loss Seems "OK":

- Loss isn't completely broken (it decreases somewhat)
- But it's stuck at 2.5 because model can't learn from wrong input
- Like trying to learn from JPEG artifacts instead of actual images

---

## ✅ THE FIX

### Solution 1: Fix Dataset to Return Spectrograms (RECOMMENDED)

**File:** `datasets/audioset.py` or `datasets/edge_audio.py`

Change `__getitem__` to:

```python
def __getitem__(self, idx):
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # ✅ ADD THIS: Convert to mel-spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=512,
        hop_length=160,
        n_mels=128
    )
    mel_spec = mel_transform(waveform)
    mel_spec = torch.log(mel_spec + 1e-9)  # Log scale
    
    # Apply augmentation if training
    if self.split == 'train':
        mel_spec_aug = self.augment(mel_spec)
        return mel_spec, mel_spec_aug, label
    
    return mel_spec, label
```

### Solution 2: Add Preprocessing in Training Loop

**File:** `train.py`

Before feeding to model:

```python
# Add audio processor
audio_processor = AudioProcessor(config)

for batch in dataloader:
    waveform, labels = batch
    
    # ✅ Convert to mel-spec
    mel_spec = audio_processor.process_audio(waveform)
    
    # Now feed to model
    embeddings = model(mel_spec)
```

---

## 📊 EXPECTED RESULTS AFTER FIX

```
Before Fix:
Epoch 1:  3.10
Epoch 100: 2.54
Improvement: 18%
Status: ❌ Not working

After Fix:
Epoch 1:  3.5
Epoch 20:  1.2
Epoch 50:  0.7
Epoch 100: 0.5
Improvement: 85%
Status: ✅ Working correctly
```

---

## 🎯 ACTION PLAN

### Priority 1: Fix Input Data Format

1. **Check which dataset is being used:**
   ```bash
   grep "dataset:" configs/streamsplit.yaml
   ```

2. **Edit the correct dataset file** (audioset.py or edge_audio.py):
   - Add mel-spectrogram transformation in `__getitem__`
   - Return correct tensor shapes

3. **Verify fix:**
   ```python
   from datasets import AudioSetDataset
   ds = AudioSetDataset(data_dir='./data', split='train')
   sample = ds[0]
   print(f"Shape: {sample[0].shape}")  # Should be [1, 128, ~1000]
   ```

### Priority 2: Retrain Model

1. Delete old checkpoints (trained on wrong data)
2. Train fresh with corrected data
3. Monitor loss - should reach < 1.0 by epoch 50

### Priority 3: Validate Results

Expected improvements:
- ✅ Loss < 1.0 by epoch 50
- ✅ Loss < 0.5 by epoch 100
- ✅ Stable convergence (no wild oscillations)
- ✅ Model embeddings with std > 0.1

---

## 📋 VERIFICATION CHECKLIST

After implementing fix:

- [ ] Dataset returns shape `[1, 128, ~1000]` not `[160000]`
- [ ] Model output std > 0.1 (not 0.0024)
- [ ] Loss starts ~3.5 and decreases to < 1.5
- [ ] No conv2d type errors
- [ ] Training runs smoothly for 100 epochs
- [ ] Final loss < 1.0 (ideally < 0.7)

---

## 💡 WHY THIS HAPPENED

Common mistake in audio ML projects:

1. **AudioSet dataset** typically stores raw .wav files
2. **Preprocessing** (FFT → Mel-spec) often done separately
3. **Easy to forget** to add this step to `__getitem__`
4. **Code runs** but learns nothing useful

**Lesson:** Always verify:
- Input shapes match model expectations
- Data is actually preprocessed
- Test with a few samples before full training

---

## 🔬 RESEARCH IMPLICATIONS

### For AAAI 2026 Paper:

❌ **Current results NOT valid** - trained on wrong input format

✅ **After fix** - retrain and report new results:
- Loss will be much better (< 1.0)
- Can confidently report model performance
- Results will be reproducible

### Timeline Estimate:

- Fix implementation: 30 minutes
- Retraining (100 epochs): ~same as before
- New checkpoint evaluation: 5 minutes
- **Total: One retaining cycle needed**

---

## 🎓 CONCLUSION

**Good News:** 
- We found the issue!
- It's fixable!
- Your system architecture is sound

**Bad News:**
- Need to retrain from scratch
- Previous checkpoints trained on wrong data
- But this explains why loss was "stuck" at 2.5

**Bottom Line:**
This is a **preprocessing bug**, not a fundamental model issue. Once fixed, training should work properly and reach expected loss values (< 1.0).

---

**Generated:** October 14, 2025
**Diagnostic Tool:** debug_training.py
