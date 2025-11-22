using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PaddleOCR.NET.ImageProcessing;
using PaddleOCR.NET.Tensor;
using SkiaSharp;
using System.Text;

namespace PaddleOCR.NET.Models.Recognition.V5;

/// <summary>
/// PP-OCRv5 Recognition Model Implementation
/// </summary>
public class RecognitionModelV5 : IRecognitionModel
{
    private readonly InferenceSession session;
    private readonly string[] characters;
    private readonly int batchSize;
    private bool disposed;

    /// <summary>
    /// Creates a new instance of the PP-OCRv5 Recognition Model
    /// </summary>
    /// <param name="modelPath">Path to the rec.onnx file</param>
    /// <param name="characterDictPath">Path to the character dictionary file (UTF-8)</param>
    /// <param name="batchSize">Batch size for processing multiple images (default: 6)</param>
    public RecognitionModelV5(
        string modelPath,
        string characterDictPath,
        int batchSize = 6)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");
        
        if (!File.Exists(characterDictPath))
            throw new FileNotFoundException($"Character dictionary file not found: {characterDictPath}");
        
        if (batchSize <= 0)
            throw new ArgumentException("Batch size must be greater than 0", nameof(batchSize));

        this.batchSize = batchSize;

        // Load character dictionary
        characters = LoadCharacterDict(characterDictPath);
        
        // Initialize ONNX session
        var sessionOptions = new SessionOptions
        {
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
        };

        session = new InferenceSession(modelPath, sessionOptions);
    }

    /// <summary>
    /// Loads character dictionary from file
    /// </summary>
    /// <param name="path">Path to character dictionary</param>
    /// <returns>Array of characters (index 0 is blank token)</returns>
    private static string[] LoadCharacterDict(string path)
    {
        var lines = File.ReadAllLines(path, Encoding.UTF8);
        
        // Prepend blank token at index 0
        var chars = new string[lines.Length + 1];
        chars[0] = ""; // Blank token
        
        for (int i = 0; i < lines.Length; i++)
        {
            chars[i + 1] = lines[i];
        }
        
        return chars;
    }

    /// <inheritdoc/>
    public RecognizedText Recognize(byte[] imageData)
    {
        using var bitmap = ImageLoader.LoadWithOrientation(imageData);
        if (bitmap == null)
            throw new InvalidOperationException("Failed to decode image");
        
        var result = RecognizeBatch([bitmap]);
        return result.Texts[0];
    }

    /// <inheritdoc/>
    public RecognitionResult RecognizeBatch(byte[][] imageBatch)
    {
        // Load all images
        var bitmaps = new SKBitmap[imageBatch.Length];
        try
        {
            for (int i = 0; i < imageBatch.Length; i++)
            {
                bitmaps[i] = ImageLoader.LoadWithOrientation(imageBatch[i])
                    ?? throw new InvalidOperationException($"Failed to decode image at index {i}");
            }
            
            return RecognizeBatch(bitmaps);
        }
        finally
        {
            // Dispose bitmaps
            foreach (var bitmap in bitmaps)
            {
                bitmap?.Dispose();
            }
        }
    }

    /// <inheritdoc/>
    public RecognitionResult RecognizeBatch(SKBitmap[] bitmaps)
    {
        if (bitmaps == null || bitmaps.Length == 0)
            throw new ArgumentException("Bitmap array cannot be null or empty", nameof(bitmaps));
        
        var allTexts = new List<RecognizedText>();
        
        // Process in batches
        for (int i = 0; i < bitmaps.Length; i += batchSize)
        {
            int currentBatchSize = Math.Min(batchSize, bitmaps.Length - i);
            var batch = new SKBitmap[currentBatchSize];
            Array.Copy(bitmaps, i, batch, 0, currentBatchSize);
            
            var batchResults = RecognizeBatchInternal(batch);
            allTexts.AddRange(batchResults);
        }
        
        return new RecognitionResult { Texts = allTexts };
    }

    /// <summary>
    /// Internal batch recognition implementation
    /// </summary>
    /// <param name="bitmaps">Array of bitmaps (size <= batchSize)</param>
    /// <returns>Array of recognized texts</returns>
    private RecognizedText[] RecognizeBatchInternal(SKBitmap[] bitmaps)
    {
        int currentBatchSize = bitmaps.Length;
        
        // Preprocess batch
        var inputTensor = RecognitionPreProcessor.PreprocessBatch(bitmaps);
        
        // Run inference
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(session.InputMetadata.First().Key, inputTensor)
        };

        using var results = session.Run(inputs);
        var output = results.First().AsTensor<float>();
        
        // Get output dimensions
        var dimensions = output.Dimensions.ToArray();
        int outputBatchSize = dimensions[0];
        int sequenceLength = dimensions[1];
        int numClasses = dimensions[2];
        
        // Convert tensor to flat array for processing
        var predictions = output.ToArray();
        
        // Decode CTC output
        var recognizedTexts = RecognitionPostProcessor.DecodeBatch(
            predictions,
            characters,
            currentBatchSize,
            sequenceLength,
            numClasses);
        
        return recognizedTexts;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (disposed) return;
        
        session?.Dispose();
        disposed = true;
        GC.SuppressFinalize(this);
    }
}
