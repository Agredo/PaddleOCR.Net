using PaddleOCR.NET.ImageProcessing;
using PaddleOCR.NET.Models.Detection.V5;

namespace PaddleOCR.NET.Examples;

/// <summary>
/// Examples demonstrating image orientation handling
/// </summary>
public static class OrientationExample
{
    /// <summary>
    /// Basic example: Detect text in images with various EXIF orientations
    /// </summary>
    public static void DetectWithOrientation()
    {
        var imagePath = @"C:\path\to\image_with_exif.jpg";
        
        using var detector = new DetectionModelV5(@"C:\path\to\det.onnx");
        
        // The image will be automatically rotated based on EXIF orientation
        var imageBytes = File.ReadAllBytes(imagePath);
        var result = detector.Detect(imageBytes);
        
        Console.WriteLine($"Detected {result.Boxes.Count} text regions");
        Console.WriteLine($"Image size after orientation correction: {result.OriginalImageSize.Width}x{result.OriginalImageSize.Height}");
    }
    
    /// <summary>
    /// Manual orientation correction: Load and correct image before detection
    /// </summary>
    public static void ManualOrientationCorrection()
    {
        var imagePath = @"C:\path\to\image_with_exif.jpg";
        
        // Manually load image with orientation correction
        using var correctedBitmap = ImageLoader.LoadWithOrientation(imagePath);
        
        Console.WriteLine($"Corrected image dimensions: {correctedBitmap.Width}x{correctedBitmap.Height}");
        
        // Now the bitmap is properly oriented for detection
        // You can save it or use it directly
    }
    
    /// <summary>
    /// Batch processing with orientation handling
    /// </summary>
    public static void BatchProcessingWithOrientation()
    {
        var imagePaths = Directory.GetFiles(@"C:\images", "*.jpg");
        
        using var detector = new DetectionModelV5Builder()
            .WithModelPath(@"C:\path\to\det.onnx")
            .WithThreshold(0.15f)
            .WithBoxThreshold(0.3f)
            .Build();
        
        foreach (var imagePath in imagePaths)
        {
            try
            {
                // Images are automatically corrected for orientation
                var imageBytes = File.ReadAllBytes(imagePath);
                var result = detector.Detect(imageBytes);
                
                Console.WriteLine($"? {Path.GetFileName(imagePath)}: {result.Boxes.Count} text regions detected");
                Console.WriteLine($"  Final dimensions: {result.OriginalImageSize.Width}x{result.OriginalImageSize.Height}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"? Error processing {Path.GetFileName(imagePath)}: {ex.Message}");
            }
        }
    }
    
    /// <summary>
    /// Compare detection results with and without orientation correction
    /// (for testing purposes - old behavior no longer available by default)
    /// </summary>
    public static void CompareOrientationHandling()
    {
        var imagePath = @"C:\path\to\rotated_image.jpg";
        
        using var detector = new DetectionModelV5(@"C:\path\to\det.onnx");
        
        // New behavior: automatic orientation correction
        var imageBytes = File.ReadAllBytes(imagePath);
        var result = detector.Detect(imageBytes);
        
        Console.WriteLine("With automatic orientation correction:");
        Console.WriteLine($"  Image size: {result.OriginalImageSize.Width}x{result.OriginalImageSize.Height}");
        Console.WriteLine($"  Detected boxes: {result.Boxes.Count}");
        
        if (result.Boxes.Count > 0)
        {
            Console.WriteLine($"  First box location: ({result.Boxes[0].Points[0].X:F0}, {result.Boxes[0].Points[0].Y:F0})");
        }
    }
    
    /// <summary>
    /// Export images with boxes - orientation is preserved
    /// </summary>
    public static void ExportWithOrientationCorrection()
    {
        var imagePath = @"C:\path\to\rotated_image.jpg";
        
        using var detector = new DetectionModelV5(@"C:\path\to\det.onnx");
        var imageBytes = File.ReadAllBytes(imagePath);
        var result = detector.Detect(imageBytes);
        
        // Export with boxes - the exported image will be in the corrected orientation
        var outputPath = ImageExporter.ExportWithBoxes(imagePath, result);
        
        Console.WriteLine($"Exported corrected image with boxes to: {outputPath}");
        Console.WriteLine("Note: The exported image is in the correct orientation (EXIF applied)");
    }
}
