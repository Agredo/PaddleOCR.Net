using PaddleOCR.NET.Models.Detection.V5;
void BasicDetection()
{
    // Create detection model
    using var detector = new DetectionModelV5(@"C:\Projects\Models\paddleocr_onnx\paddleocr-onnx\detection\v5\det.onnx");

    // Load image
    var imageBytes = File.ReadAllBytes(@"C:\Users\chris\Downloads\Test.png");

    // Detect text regions
    var result = detector.Detect(imageBytes);

    // Process results
    Console.WriteLine($"Original Size: {result.OriginalImageSize.Width}x{result.OriginalImageSize.Height}");
    Console.WriteLine($"Processed Size: {result.ProcessedImageSize.Width}x{result.ProcessedImageSize.Height}");
    Console.WriteLine($"Found {result.Boxes.Count} text regions\n");

    foreach (var box in result.Boxes)
    {
        Console.WriteLine($"Confidence: {box.Confidence:F2}");
        for (int i = 0; i < box.Points.Length; i++)
        {
            Console.WriteLine($"  Point {i + 1}: ({box.Points[i].X:F1}, {box.Points[i].Y:F1})");
        }
        Console.WriteLine();
    }
}

BasicDetection();
