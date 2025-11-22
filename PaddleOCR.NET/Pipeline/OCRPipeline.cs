using PaddleOCR.NET.Models.Detection;
using PaddleOCR.NET.Models.Recognition;
using PaddleOCR.NET.ImageProcessing;
using SkiaSharp;

namespace PaddleOCR.NET.Pipeline;

/// <summary>
/// Complete OCR pipeline combining detection and recognition
/// </summary>
public class OCRPipeline : IDisposable
{
    private readonly IDetectionModel detectionModel;
    private readonly IRecognitionModel recognitionModel;
    private bool disposed;

    /// <summary>
    /// Creates a new OCR pipeline
    /// </summary>
    /// <param name="detectionModel">Detection model instance</param>
    /// <param name="recognitionModel">Recognition model instance</param>
    public OCRPipeline(IDetectionModel detectionModel, IRecognitionModel recognitionModel)
    {
        this.detectionModel = detectionModel ?? throw new ArgumentNullException(nameof(detectionModel));
        this.recognitionModel = recognitionModel ?? throw new ArgumentNullException(nameof(recognitionModel));
    }

    /// <summary>
    /// Processes an image and returns OCR results
    /// </summary>
    /// <param name="imageData">Image data as byte array</param>
    /// <returns>OCR result with detected regions and recognized text</returns>
    public OCRResult Process(byte[] imageData)
    {
        if (imageData == null || imageData.Length == 0)
            throw new ArgumentException("Image data cannot be null or empty", nameof(imageData));

        // Step 1: Detect text regions
        var detectionResult = detectionModel.Detect(imageData);
        
        if (detectionResult.Boxes.Count == 0)
        {
            return new OCRResult
            {
                DetectionResult = detectionResult,
                TextRegions = Array.Empty<OCRTextRegion>()
            };
        }

        // Step 2: Load the original image
        using var originalBitmap = ImageLoader.LoadWithOrientation(imageData);
        if (originalBitmap == null)
            throw new InvalidOperationException("Failed to load image");

        // Step 3: Crop all detected regions
        var croppedBitmaps = ImageCropper.CropRegions(originalBitmap, detectionResult.Boxes);

        try
        {
            // Step 4: Recognize text in all regions (batch processing)
            var recognitionResult = recognitionModel.RecognizeBatch(croppedBitmaps);

            // Step 5: Combine detection and recognition results
            var textRegions = new List<OCRTextRegion>();
            
            for (int i = 0; i < detectionResult.Boxes.Count; i++)
            {
                textRegions.Add(new OCRTextRegion
                {
                    BoundingBox = detectionResult.Boxes[i],
                    Text = recognitionResult.Texts[i],
                    Index = i
                });
            }

            return new OCRResult
            {
                DetectionResult = detectionResult,
                TextRegions = textRegions
            };
        }
        finally
        {
            // Dispose cropped bitmaps
            foreach (var bitmap in croppedBitmaps)
            {
                bitmap?.Dispose();
            }
        }
    }

    /// <summary>
    /// Processes multiple images
    /// </summary>
    /// <param name="imageBatch">Array of image data</param>
    /// <returns>Array of OCR results</returns>
    public OCRResult[] ProcessBatch(byte[][] imageBatch)
    {
        if (imageBatch == null || imageBatch.Length == 0)
            throw new ArgumentException("Image batch cannot be null or empty", nameof(imageBatch));

        var results = new OCRResult[imageBatch.Length];
        
        for (int i = 0; i < imageBatch.Length; i++)
        {
            results[i] = Process(imageBatch[i]);
        }
        
        return results;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (disposed) return;
        
        // Note: We don't dispose the models here as they might be reused
        // The caller is responsible for disposing the models
        
        disposed = true;
        GC.SuppressFinalize(this);
    }
}
