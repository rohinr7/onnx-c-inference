using Microsoft.ML.OnnxRuntime;
using OnnxRuntime.ResNet.Template.utils;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace OnnxRuntime.ResNet.Template
{
    class Program
    {
        public static void Main(string[] args)
        {
            // Read paths
            string modelFilePath = @"C:\Users\rohin\source\repos\onnxmar2\model\resnet_tvs.onnx";
            string imageFilePath = @"C:\Users\rohin\source\repos\onnxmar2\data\frame_000031.PNG";
            Stopwatch stopwatch = new Stopwatch();
            
            var input = ImageHelper.GetImageTensorFromPath(imageFilePath);

            stopwatch.Start();
            
            var top10 = ModelHelper.GetPredictions(input, modelFilePath);

            stopwatch.Stop();
            

            // Print results to console
            Console.WriteLine("Top 10 predictions for ResNet50 v2...");
            Console.WriteLine("--------------------------------------------------------------");
            //Console.WriteLine($"Label: {top10[0].Label}, Confidence: {top10[0].Confidence}");
            foreach (var t in top10)
            {
                Console.WriteLine($"Label: {t.Label}, Confidence: {t.Confidence}");

            }
            Console.WriteLine("Elapsed Time is {0}ms", stopwatch.ElapsedMilliseconds);
        }
    }
}
