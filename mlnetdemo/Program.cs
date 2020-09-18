using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Transforms.Onnx;

namespace mlnetdemo
{
    class Program
    {
        public const int ImageHeight = 28;
        public const int ImageWidth = 28;
        public static string GetAbsolutePath(string relativePath)
        {
            var _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            var assemblyFolderPath = _dataRoot.Directory.FullName;

            var fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        static void Main(string[] args)
        {
            var assetsRelativePath = @"../../../assets";
            var assetsPath = GetAbsolutePath(assetsRelativePath);
            var modelFilePath = Path.Combine(assetsPath, "model-2.onnx");
            var imagesFolder = Path.Combine(assetsPath, "images");
            var outputFolder = Path.Combine(assetsPath, "images", "output");

            var mlContext = new MLContext();
            
            var images = ImageNetData.ReadFromFile(imagesFolder);
            var imageDataView = mlContext.Data.LoadFromEnumerable(images);

            var model = LoadModel(mlContext,modelFilePath);

            

            var scoredData = model.Transform(imageDataView);
          
            var probabilities = scoredData.GetColumn<float[]>("output");

            var data = probabilities.First();
            Console.WriteLine("========= End of Process..Hit any Key ========");
            Console.ReadLine();
        }

        private static ITransformer LoadModel(MLContext mlContext,string modelLocation)
        {
            Console.WriteLine("Read model");
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Default parameters: image size=({ImageWidth},{ImageHeight})");

            // Create IDataView from empty list to obtain input data schema
            var data = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());

            // Define scoring pipeline
            var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "image", imageFolder: "", inputColumnName: nameof(ImageNetData.ImagePath))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "image", imageWidth: ImageWidth, imageHeight: ImageHeight, inputColumnName: "image"))
                .Append(mlContext.Transforms.ConvertToGrayscale("image"))
                .Append(mlContext.Transforms.ExtractPixels(inputColumnName:"image", outputColumnName: "input",colorsToExtract:ImagePixelExtractingEstimator.ColorBits.Red))
                .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: modelLocation, outputColumnNames: new[] { "output" }, inputColumnNames: new[] { "input" }));

            // Fit scoring pipeline
            var model = pipeline.Fit(data);

            return model;
        }

        public class ImageNetData
        {
            [LoadColumn(0)]
            public string ImagePath;

            [LoadColumn(1)]
            public string Label;

            public static IEnumerable<ImageNetData> ReadFromFile(string imageFolder)
            {
                return Directory
                    .GetFiles(imageFolder)
                    .Where(filePath => Path.GetExtension(filePath) != ".md")
                    .Select(filePath => new ImageNetData { ImagePath = filePath, Label = Path.GetFileName(filePath) });
            }
        }
    }
}
