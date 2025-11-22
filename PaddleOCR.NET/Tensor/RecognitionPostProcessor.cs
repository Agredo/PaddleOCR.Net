using PaddleOCR.NET.Models.Recognition;

namespace PaddleOCR.NET.Tensor;

/// <summary>
/// Post-processing for PP-OCR Recognition models using CTC decoding
/// </summary>
public static class RecognitionPostProcessor
{
    /// <summary>
    /// Decodes CTC output to recognized text using greedy decoding
    /// </summary>
    /// <param name="predictions">Model output tensor [batch_size, sequence_length, num_classes]</param>
    /// <param name="characters">Character dictionary (index 0 must be blank token)</param>
    /// <param name="batchSize">Number of images in batch</param>
    /// <param name="sequenceLength">Sequence length dimension</param>
    /// <param name="numClasses">Number of character classes</param>
    /// <returns>Array of recognized texts</returns>
    public static RecognizedText[] DecodeBatch(
        float[] predictions,
        string[] characters,
        int batchSize,
        int sequenceLength,
        int numClasses)
    {
        var results = new RecognizedText[batchSize];
        
        for (int b = 0; b < batchSize; b++)
        {
            results[b] = DecodeSingle(predictions, characters, b, sequenceLength, numClasses);
        }
        
        return results;
    }
    
    /// <summary>
    /// Decodes CTC output for a single image
    /// </summary>
    /// <param name="predictions">Model output tensor</param>
    /// <param name="characters">Character dictionary</param>
    /// <param name="batchIndex">Index in the batch</param>
    /// <param name="sequenceLength">Sequence length dimension</param>
    /// <param name="numClasses">Number of character classes</param>
    /// <returns>Recognized text with confidence</returns>
    private static RecognizedText DecodeSingle(
        float[] predictions,
        string[] characters,
        int batchIndex,
        int sequenceLength,
        int numClasses)
    {
        // Extract predictions for this batch item
        var indices = new int[sequenceLength];
        var probabilities = new float[sequenceLength];
        
        // Get argmax (character index) and max probability for each time step
        int offset = batchIndex * sequenceLength * numClasses;
        
        for (int t = 0; t < sequenceLength; t++)
        {
            int timeOffset = offset + t * numClasses;
            int maxIdx = 0;
            float maxProb = predictions[timeOffset];
            
            for (int c = 1; c < numClasses; c++)
            {
                float prob = predictions[timeOffset + c];
                if (prob > maxProb)
                {
                    maxProb = prob;
                    maxIdx = c;
                }
            }
            
            indices[t] = maxIdx;
            probabilities[t] = maxProb;
        }
        
        // Apply CTC decoding rules:
        // 1. Remove consecutive duplicates
        // 2. Remove blank tokens (index 0)
        var charList = new List<string>();
        var confList = new List<float>();
        
        int? previousIdx = null;
        
        for (int t = 0; t < sequenceLength; t++)
        {
            int currentIdx = indices[t];
            
            // Skip if same as previous (remove consecutive duplicates)
            if (previousIdx.HasValue && currentIdx == previousIdx.Value)
                continue;
            
            // Skip blank token (index 0)
            if (currentIdx == 0)
            {
                previousIdx = currentIdx;
                continue;
            }
            
            // Add character if valid
            if (currentIdx > 0 && currentIdx < characters.Length)
            {
                charList.Add(characters[currentIdx]);
                confList.Add(probabilities[t]);
            }
            
            previousIdx = currentIdx;
        }
        
        // Build final text and calculate confidence
        var text = string.Join("", charList);
        var confidence = confList.Count > 0 ? confList.Average() : 0f;
        var charConfidences = confList.ToArray();
        
        return new RecognizedText(text, confidence, charConfidences);
    }
}
