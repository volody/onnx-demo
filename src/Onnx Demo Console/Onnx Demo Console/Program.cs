using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Onnx_Demo_Console
{
    public class Score
    {
        public int Prediction { get; set; }
        public double Time { get; set; }
        public List<double> Scores { get; set; }
        public string Status { get; set; }
        public bool Empty { get; set; } = true;
    }

    class Program
    {
        static void Main(string[] args)
        {
            // 1. download model
            // https://www.tensorflow.org/lite/models/bert_qa/overview
            // download model
            // https://storage.googleapis.com/download.tensorflow.org/models/tflite/bert_qa/mobilebert_qa_vocab.zip


            // 2. convert to onnx

            // 3. run inverence
            //var session = new InferenceSession("model.onnx");

            // 4. display result
        }

        private Score PredictLocal(InferenceSession session, float[] digit)
        {
            var now = DateTime.Now;
            Tensor<float> x = new DenseTensor<float>(digit.Length);

            for (int i = 0; i <= digit.Length - 1; i++)
                x[i] = digit[i] / 255.0f;

            var input = new List<NamedOnnxValue>()
            {
                NamedOnnxValue.CreateFromTensor("0", x)
            };

            try
            {
                var prediction = session.Run(input).First().AsTensor<float>().ToArray();
                return new Score()
                {
                    Status = $"Local Mode: {session}",
                    Empty = false,
                    Prediction = Array.IndexOf(prediction, prediction.Max()),
                    Scores = prediction.Select(i => System.Convert.ToDouble(i)).ToList(),
                    Time = (DateTime.Now - now).TotalSeconds
                };
            }
            catch (Exception e)
            {
                return new Score()
                {
                    Status = e.Message
                };
            }
        }

    }
}
