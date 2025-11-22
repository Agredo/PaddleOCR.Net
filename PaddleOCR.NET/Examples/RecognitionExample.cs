using PaddleOCR.NET.Models.Recognition.V5;
using PaddleOCR.NET.Models.Detection.V5;

namespace PaddleOCR.NET.Examples;

/// <summary>
/// Example usage of PP-OCR Recognition Model
/// </summary>
public static class RecognitionExample
{
    /// <summary>
    /// Example: Recognize text in a single image
    /// </summary>
    public static void RecognizeSingleImage()
    {
        using var recognizer = new RecognitionModelV5Builder()
            .WithModelPath("path/to/rec.onnx")
            .WithCharacterDict("path/to/ppocr_keys_v1.txt")
            .WithBatchSize(6)
            .Build();
        
        var imageBytes = File.ReadAllBytes("path/to/text_image.jpg");
        var result = recognizer.Recognize(imageBytes);
        
        Console.WriteLine($"Recognized Text: {result.Text}");
        Console.WriteLine($"Confidence: {result.Confidence:F3}");
    }
    
    /// <summary>
    /// Example: Recognize text in multiple images (batch processing)
    /// </summary>
    public static void RecognizeBatch()
    {
        using var recognizer = new RecognitionModelV5Builder()
            .WithModelPath("path/to/rec.onnx")
            .WithCharacterDict("path/to/ppocr_keys_v1.txt")
            .WithBatchSize(6)  // Process 6 images at a time
            .Build();
        
        var imagePaths = new[]
        {
            "path/to/image1.jpg",
            "path/to/image2.jpg",
            "path/to/image3.jpg"
        };
        
        var imageBatch = imagePaths.Select(File.ReadAllBytes).ToArray();
        var results = recognizer.RecognizeBatch(imageBatch);
        
        Console.WriteLine($"Recognized {results.Count} texts:");
        for (int i = 0; i < results.Count; i++)
        {
            var text = results.Texts[i];
            Console.WriteLine($"{i + 1}. \"{text.Text}\" (confidence: {text.Confidence:F3})");
        }
    }
    
    /// <summary>
    /// Example: Complete OCR pipeline (Detection + Recognition)
    /// </summary>
    public static void CompleteOCRPipeline()
    {
        // Step 1: Detect text regions
        using var detector = new DetectionModelV5Builder()
            .WithModelPath("path/to/det.onnx")
            .WithThreshold(0.3f)
            .WithBoxThreshold(0.5f)
            .Build();
        
        // Step 2: Setup recognition model
        using var recognizer = new RecognitionModelV5Builder()
            .WithModelPath("path/to/rec.onnx")
            .WithCharacterDict("path/to/ppocr_keys_v1.txt")
            .Build();
        
        // Step 3: Process image
        var imageBytes = File.ReadAllBytes("path/to/document.jpg");
        var detectionResult = detector.Detect(imageBytes);
        
        Console.WriteLine($"Found {detectionResult.Boxes.Count} text regions");
        
        // TODO: Extract cropped regions from bounding boxes and recognize text
        // This would require additional image cropping functionality
        
        foreach (var box in detectionResult.Boxes)
        {
            Console.WriteLine($"Box at ({box.Points[0].X:F1}, {box.Points[0].Y:F1}) " +
                            $"with confidence {box.Confidence:F3}");
            // In a complete implementation, you would:
            // 1. Crop the image region using the bounding box
            // 2. Pass it to recognizer.Recognize()
            // 3. Display the recognized text
        }
    }
}
