using PaddleOCR.NET.Models.Detection.V5;
using PaddleOCR.NET.Models.Recognition.V5;
using PaddleOCR.NET.ImageProcessing;
using PaddleOCR.NET.Pipeline;

void TestDetection()
{
    Console.WriteLine("=== PaddleOCR Detection Test ===\n");
    
    var imagePath = @"C:\Users\chris\OneDrive\Bilder\Camera Roll\IMG_20250108_024734.jpg";
    
    // Test with very low thresholds first
    using var detector = new DetectionModelV5Builder()
        .WithModelPath(@"C:\Projects\Models\paddleocr_onnx\paddleocr-onnx\detection\v5\det.onnx")
        .WithThreshold(0.3f)      // Very low for debugging
        .WithBoxThreshold(0.5f)   // Very low for debugging
        .WithUnclipRatio(1.2f)
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

void TestDetectionWithMerging()
{
    Console.WriteLine("=== PaddleOCR Detection Test with Merging ===\n");
    
    var imagePath = @"C:\Users\chris\OneDrive\Bilder\Camera Roll\IMG_20250108_024734.jpg";
    
    // Test with box merging enabled
    using var detector = new DetectionModelV5Builder()
        .WithModelPath(@"C:\Projects\Models\paddleocr_onnx\paddleocr-onnx\detection\v5\det.onnx")
        .WithThreshold(0.3f)
        .WithBoxThreshold(0.5f)
        .WithUnclipRatio(1.6f)
        .WithBoxMerging(true)                    // Enable box merging
        .WithMergeDistanceThreshold(0.01f)        // Merge boxes within 0.5x box height distance
        

        .Build();

    var imageBytes = File.ReadAllBytes(imagePath);
    
    Console.WriteLine("Processing image with box merging enabled...\n");
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
                outputSuffix: "_detected_merged",
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

void TestRecognition()
{
    Console.WriteLine("\n=== PaddleOCR Recognition Test ===\n");
    
    // Example paths (adjust to your setup)
    var modelPath = @"C:\Projects\Models\paddleocr_onnx\paddleocr-onnx\languages\latin\rec.onnx";
    var charDictPath = @"C:\Projects\Models\paddleocr_onnx\paddleocr-onnx\languages\latin\dict.txt";
    var testImagePath = @"C:\Users\chris\OneDrive\Bilder\Camera Roll\IMG_20250108_024734.jpg";
    
    // Check if files exist before running
    if (!File.Exists(modelPath))
    {
        Console.WriteLine($"⚠️ Recognition model not found at: {modelPath}");
        Console.WriteLine("   Please update the path to your rec.onnx file");
        return;
    }
    
    if (!File.Exists(charDictPath))
    {
        Console.WriteLine($"⚠️ Character dictionary not found at: {charDictPath}");
        Console.WriteLine("   Please update the path to your character dictionary file");
        return;
    }
    
    if (!File.Exists(testImagePath))
    {
        Console.WriteLine($"⚠️ Test image not found at: {testImagePath}");
        Console.WriteLine("   Please update the path to a text image file");
        return;
    }
    
    try
    {
        // Build recognition model
        using var recognizer = new RecognitionModelV5Builder()
            .WithModelPath(modelPath)
            .WithCharacterDict(charDictPath)
            .WithBatchSize(6)
            .Build();
        
        // Test single image recognition
        Console.WriteLine("Processing single image...");
        var imageBytes = File.ReadAllBytes(testImagePath);
        var result = recognizer.Recognize(imageBytes);
        
        Console.WriteLine($"\n=== RESULTS ===");
        Console.WriteLine($"Text: {result.Text}");
        Console.WriteLine($"Confidence: {result.Confidence:F3}");
        
        if (result.CharConfidences != null && result.CharConfidences.Length > 0)
        {
            Console.WriteLine($"Character confidences:");
            for (int i = 0; i < Math.Min(result.Text.Length, result.CharConfidences.Length); i++)
            {
                Console.WriteLine($"  '{result.Text[i]}': {result.CharConfidences[i]:F3}");
            }
        }
        
        Console.WriteLine("\n✅ Recognition test completed successfully!");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"\n✗ Recognition test failed: {ex.Message}");
        Console.WriteLine($"   Stack trace: {ex.StackTrace}");
    }
}

void TestCompleteOCRPipeline()
{
    Console.WriteLine("\n=== Complete OCR Pipeline Test ===\n");
    
    // Paths
    var imagePath = @"C:\Users\chris\OneDrive\Bilder\Camera Roll\IMG_20250108_024734.jpg";
    var detModelPath = @"C:\Projects\Models\paddleocr_onnx\paddleocr-onnx\detection\v5\det.onnx";
    var recModelPath = @"C:\Projects\Models\paddleocr_onnx\paddleocr-onnx\languages\latin\rec.onnx";
    var charDictPath = @"C:\Projects\Models\paddleocr_onnx\paddleocr-onnx\languages\latin\dict.txt";
    
    // Check if files exist
    if (!File.Exists(detModelPath))
    {
        Console.WriteLine($"⚠️ Detection model not found at: {detModelPath}");
        return;
    }
    
    if (!File.Exists(recModelPath))
    {
        Console.WriteLine($"⚠️ Recognition model not found at: {recModelPath}");
        return;
    }
    
    if (!File.Exists(charDictPath))
    {
        Console.WriteLine($"⚠️ Character dictionary not found at: {charDictPath}");
        return;
    }
    
    if (!File.Exists(imagePath))
    {
        Console.WriteLine($"⚠️ Test image not found at: {imagePath}");
        return;
    }
    
    try
    {
        Console.WriteLine("Building OCR pipeline...");
        
        // Build complete OCR pipeline
        using var pipeline = new OCRPipelineBuilder()
            .WithDetection(builder => builder
                .WithModelPath(detModelPath)
                .WithThreshold(0.3f)
                .WithBoxThreshold(0.5f)
                .WithUnclipRatio(1.6f))
            .WithRecognition(builder => builder
                .WithModelPath(recModelPath)
                .WithCharacterDict(charDictPath)
                .WithBatchSize(6))
            .Build();
        
        Console.WriteLine("Processing image...\n");
        
        // Process the image
        var imageBytes = File.ReadAllBytes(imagePath);
        var result = pipeline.Process(imageBytes);
        
        // Display results
        Console.WriteLine($"=== RESULTS ===");
        Console.WriteLine($"Image Size: {result.DetectionResult.OriginalImageSize.Width}x{result.DetectionResult.OriginalImageSize.Height}");
        Console.WriteLine($"Found {result.Count} text regions\n");
        
        if (result.Count == 0)
        {
            Console.WriteLine("⚠️ No text detected in the image");
            return;
        }
        
        // Sort regions by position (top to bottom, left to right)
        var sortedRegions = result.TextRegions
            .OrderBy(r => r.BoundingBox.Points[0].Y)
            .ThenBy(r => r.BoundingBox.Points[0].X)
            .ToList();
        
        // Display each region
        foreach (var region in sortedRegions)
        {
            Console.WriteLine($"Region {region.Index + 1}:");
            Console.WriteLine($"  Text: \"{region.Text.Text}\"");
            Console.WriteLine($"  Recognition Confidence: {region.Text.Confidence:F3}");
            Console.WriteLine($"  Detection Confidence: {region.BoundingBox.Confidence:F3}");
            Console.WriteLine($"  Position: ({region.BoundingBox.Points[0].X:F0}, {region.BoundingBox.Points[0].Y:F0})");
            Console.WriteLine();
        }
        
        // Display full text
        Console.WriteLine("=== FULL TEXT ===");
        Console.WriteLine(result.GetFullText("\n"));
        Console.WriteLine();
        
        // Statistics
        var avgDetectionConf = result.TextRegions.Average(r => r.BoundingBox.Confidence);
        var avgRecognitionConf = result.TextRegions.Average(r => r.Text.Confidence);
        var totalChars = result.TextRegions.Sum(r => r.Text.Text.Length);
        
        Console.WriteLine("=== STATISTICS ===");
        Console.WriteLine($"Average Detection Confidence: {avgDetectionConf:F3}");
        Console.WriteLine($"Average Recognition Confidence: {avgRecognitionConf:F3}");
        Console.WriteLine($"Total Characters: {totalChars}");
        
        // Export results
        Console.WriteLine("\n=== EXPORTING RESULTS ===");
        try
        {
            var outputPath = ImageExporter.ExportWithBoxes(
                imagePath,
                result.DetectionResult,
                outputSuffix: "_ocr",
                strokeWidth: 3f,
                showConfidence: true,
                quality: 95
            );
            
            Console.WriteLine($"✓ Image with boxes exported:");
            Console.WriteLine($"  {outputPath}");
            
            // Export text to file
            var textOutputPath = Path.ChangeExtension(outputPath, ".txt");
            File.WriteAllText(textOutputPath, result.GetFullText("\n"));
            Console.WriteLine($"✓ Text exported:");
            Console.WriteLine($"  {textOutputPath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"✗ Failed to export results: {ex.Message}");
        }
        
        Console.WriteLine("\n✅ Complete OCR pipeline test completed successfully!");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"\n✗ OCR pipeline test failed: {ex.Message}");
        Console.WriteLine($"   Stack trace: {ex.StackTrace}");
    }
}

TestDetection();
//TestDetectionWithMerging();

// Uncomment to test recognition (requires rec.onnx model and character dictionary)
// TestRecognition();

// Uncomment to test complete OCR pipeline (requires both detection and recognition models)
 //TestCompleteOCRPipeline();
