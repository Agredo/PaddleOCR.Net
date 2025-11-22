using PaddleOCR.NET.Models.Detection.V5;
using PaddleOCR.NET.ImageProcessing;

void TestDetection()
{
    Console.WriteLine("=== PaddleOCR Detection Test ===\n");
    
    var imagePath = @"C:\Users\chris\OneDrive\Bilder\Camera Roll\WIN_20250108_02_38_22_Pro.jpg";
    
    // Test with very low thresholds first
    using var detector = new DetectionModelV5Builder()
        .WithModelPath(@"C:\Projects\Models\paddleocr_onnx\paddleocr-onnx\detection\v5\det.onnx")
        .WithThreshold(0.3f)      // Very low for debugging
        .WithBoxThreshold(0.5f)   // Very low for debugging
        .WithUnclipRatio(1.6f)
        .Build();

    var imageBytes = File.ReadAllBytes(imagePath);
    
    Console.WriteLine("Processing image...\n");
    var result = detector.Detect(imageBytes);

    Console.WriteLine($"\n=== RESULTS ===");
    Console.WriteLine($"Original Size: {result.OriginalImageSize.Width}x{result.OriginalImageSize.Height}");
    Console.WriteLine($"Processed Size: {result.ProcessedImageSize.Width}x{result.ProcessedImageSize.Height}");
    Console.WriteLine($"Found {result.Boxes.Count} text regions\n");

    if (result.Boxes.Count == 0)
    {
        Console.WriteLine("⚠️ No boxes detected! Check the debug output above.");
        Console.WriteLine("Suggestions:");
        Console.WriteLine("  1. Try even lower thresholds (0.05)");
        Console.WriteLine("  2. Verify the ONNX model is correct for PP-OCRv5");
        Console.WriteLine("  3. Check if the image contains detectable text");
    }
    else
    {
        // Validate coordinates
        var allValid = true;
        foreach (var box in result.Boxes)
        {
            Console.WriteLine($"Box: Confidence={box.Confidence:F3}");
            for (int i = 0; i < box.Points.Length; i++)
            {
                var point = box.Points[i];
                var isValid = point.X >= 0 && point.X <= result.OriginalImageSize.Width &&
                              point.Y >= 0 && point.Y <= result.OriginalImageSize.Height;
                
                var status = isValid ? "✓" : "✗";
                Console.WriteLine($"  Point {i + 1}: ({point.X:F1}, {point.Y:F1}) {status}");
                
                if (!isValid)
                    allValid = false;
            }
            Console.WriteLine();
        }
        
        if (allValid)
        {
            Console.WriteLine("✅ All coordinates are within image bounds!");
        }
        else
        {
            Console.WriteLine("⚠️ Some coordinates are outside image bounds!");
        }
        
        // Export image with bounding boxes
        Console.WriteLine("\n=== EXPORTING IMAGE ===");
        try
        {
            var outputPath = ImageExporter.ExportWithBoxes(
                imagePath,
                result,
                outputSuffix: "_detected",
                strokeWidth: 3f,
                showConfidence: true,
                quality: 95
            );
            
            Console.WriteLine($"✓ Image exported successfully:");
            Console.WriteLine($"  {outputPath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"✗ Failed to export image: {ex.Message}");
        }
    }
}

TestDetection();
