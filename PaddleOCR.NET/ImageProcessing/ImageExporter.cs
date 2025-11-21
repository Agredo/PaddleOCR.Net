using PaddleOCR.NET.Models.Detection;
using SkiaSharp;

namespace PaddleOCR.NET.ImageProcessing;

/// <summary>
/// Exports images with visualized detection results
/// </summary>
public static class ImageExporter
{
    /// <summary>
    /// Exports an image with drawn bounding boxes to the same directory
    /// </summary>
    /// <param name="originalImagePath">Path to the original image</param>
    /// <param name="detectionResult">Detection result containing bounding boxes</param>
    /// <param name="outputSuffix">Suffix to append to the filename (default: "_detected")</param>
    /// <param name="strokeWidth">Width of the bounding box lines (default: 3)</param>
    /// <param name="showConfidence">Whether to display confidence values (default: true)</param>
    /// <param name="quality">JPEG quality (1-100, default: 95)</param>
    /// <returns>Path to the exported image</returns>
    public static string ExportWithBoxes(
        string originalImagePath,
        DetectionResult detectionResult,
        string outputSuffix = "_detected",
        float strokeWidth = 3f,
        bool showConfidence = true,
        int quality = 95)
    {
        if (!File.Exists(originalImagePath))
            throw new FileNotFoundException($"Image file not found: {originalImagePath}");

        // Generate output path
        var directory = Path.GetDirectoryName(originalImagePath) ?? string.Empty;
        var fileNameWithoutExt = Path.GetFileNameWithoutExtension(originalImagePath);
        var extension = Path.GetExtension(originalImagePath);
        var outputPath = Path.Combine(directory, $"{fileNameWithoutExt}{outputSuffix}{extension}");

        // Load the original image with orientation correction
        using var bitmap = ImageLoader.LoadWithOrientation(originalImagePath);
        if (bitmap == null)
            throw new InvalidOperationException($"Failed to decode image: {originalImagePath}");

        // Create a surface to draw on
        using var surface = SKSurface.Create(new SKImageInfo(bitmap.Width, bitmap.Height));
        var canvas = surface.Canvas;

        // Draw the original image
        canvas.DrawBitmap(bitmap, 0, 0);

        // Draw bounding boxes
        DrawBoundingBoxes(canvas, detectionResult.Boxes, strokeWidth, showConfidence);

        // Save the image
        using var image = surface.Snapshot();
        using var data = extension.ToLowerInvariant() switch
        {
            ".png" => image.Encode(SKEncodedImageFormat.Png, 100),
            ".jpg" or ".jpeg" => image.Encode(SKEncodedImageFormat.Jpeg, quality),
            ".webp" => image.Encode(SKEncodedImageFormat.Webp, quality),
            _ => image.Encode(SKEncodedImageFormat.Png, 100)
        };

        using var stream = File.OpenWrite(outputPath);
        data.SaveTo(stream);

        return outputPath;
    }

    /// <summary>
    /// Exports an image with drawn bounding boxes from byte array
    /// </summary>
    /// <param name="imageData">Original image data</param>
    /// <param name="outputPath">Path where to save the output image</param>
    /// <param name="detectionResult">Detection result containing bounding boxes</param>
    /// <param name="strokeWidth">Width of the bounding box lines (default: 3)</param>
    /// <param name="showConfidence">Whether to display confidence values (default: true)</param>
    /// <param name="format">Output image format (default: PNG)</param>
    /// <param name="quality">JPEG/WebP quality (1-100, default: 95)</param>
    /// <returns>Path to the exported image</returns>
    public static string ExportWithBoxes(
        byte[] imageData,
        string outputPath,
        DetectionResult detectionResult,
        float strokeWidth = 3f,
        bool showConfidence = true,
        SKEncodedImageFormat format = SKEncodedImageFormat.Png,
        int quality = 95)
    {
        // Load the original image with orientation correction
        using var bitmap = ImageLoader.LoadWithOrientation(imageData);
        if (bitmap == null)
            throw new InvalidOperationException("Failed to decode image data");

        // Create a surface to draw on
        using var surface = SKSurface.Create(new SKImageInfo(bitmap.Width, bitmap.Height));
        var canvas = surface.Canvas;

        // Draw the original image
        canvas.DrawBitmap(bitmap, 0, 0);

        // Draw bounding boxes
        DrawBoundingBoxes(canvas, detectionResult.Boxes, strokeWidth, showConfidence);

        // Save the image
        using var image = surface.Snapshot();
        using var data = image.Encode(format, quality);

        // Ensure directory exists
        var directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            Directory.CreateDirectory(directory);

        using var stream = File.OpenWrite(outputPath);
        data.SaveTo(stream);

        return outputPath;
    }

    /// <summary>
    /// Draws bounding boxes on a canvas
    /// </summary>
    private static void DrawBoundingBoxes(
        SKCanvas canvas,
        IReadOnlyList<BoundingBox> boxes,
        float strokeWidth,
        bool showConfidence)
    {
        // Generate distinct colors for each box
        var colors = GenerateDistinctColors(boxes.Count);

        for (int i = 0; i < boxes.Count; i++)
        {
            var box = boxes[i];
            var color = colors[i];

            // Draw the polygon connecting all points
            using var paint = new SKPaint
            {
                Color = color,
                StrokeWidth = strokeWidth,
                Style = SKPaintStyle.Stroke,
                IsAntialias = true
            };

            using var path = new SKPath();
            path.MoveTo(box.Points[0].X, box.Points[0].Y);
            for (int j = 1; j < box.Points.Length; j++)
            {
                path.LineTo(box.Points[j].X, box.Points[j].Y);
            }
            path.Close();

            canvas.DrawPath(path, paint);

            // Draw corner circles for better visibility
            using var circlePaint = new SKPaint
            {
                Color = color,
                Style = SKPaintStyle.Fill,
                IsAntialias = true
            };

            foreach (var point in box.Points)
            {
                canvas.DrawCircle(point.X, point.Y, strokeWidth * 1.5f, circlePaint);
            }

            // Draw confidence text if requested
            if (showConfidence)
            {
                var confidence = $"{box.Confidence:P0}";
                var textPoint = box.Points[0];

                // Background for text
                using var textPaint = new SKPaint
                {
                    Color = SKColors.White,
                    TextSize = 16,
                    IsAntialias = true,
                    Typeface = SKTypeface.FromFamilyName("Arial", SKFontStyle.Bold)
                };

                var textBounds = new SKRect();
                textPaint.MeasureText(confidence, ref textBounds);

                var backgroundRect = new SKRect(
                    textPoint.X - 2,
                    textPoint.Y - textBounds.Height - 4,
                    textPoint.X + textBounds.Width + 4,
                    textPoint.Y + 2
                );

                using var backgroundPaint = new SKPaint
                {
                    Color = color.WithAlpha(200),
                    Style = SKPaintStyle.Fill
                };

                canvas.DrawRect(backgroundRect, backgroundPaint);
                canvas.DrawText(confidence, textPoint.X, textPoint.Y, textPaint);
            }
        }
    }

    /// <summary>
    /// Generates a list of visually distinct colors
    /// </summary>
    private static List<SKColor> GenerateDistinctColors(int count)
    {
        var colors = new List<SKColor>();

        if (count == 0)
            return colors;

        // Predefined distinct colors
        var baseColors = new[]
        {
            SKColors.Red,
            SKColors.Green,
            SKColors.Blue,
            SKColors.Orange,
            SKColors.Purple,
            SKColors.Cyan,
            SKColors.Magenta,
            SKColors.Yellow,
            SKColors.Lime,
            SKColors.Pink,
            SKColors.Teal,
            SKColors.Navy,
            SKColors.Maroon,
            SKColors.Olive,
            SKColors.Coral
        };

        // Use predefined colors if we have enough
        if (count <= baseColors.Length)
        {
            colors.AddRange(baseColors.Take(count));
        }
        else
        {
            // Generate additional colors using HSV
            for (int i = 0; i < count; i++)
            {
                var hue = (i * 360f / count) % 360f;
                var saturation = 0.7f + (i % 3) * 0.1f;
                var value = 0.8f + (i % 2) * 0.1f;

                colors.Add(SKColor.FromHsv(hue, saturation * 100, value * 100));
            }
        }

        return colors;
    }
}
