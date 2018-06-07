-- Copyright 2017 Elkin Cruz.
-- Copyright 2017 The TensorFlow Authors.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

{-# LANGUAGE ScopedTypeVariables, PartialTypeSignatures  #-}
{-# OPTIONS_GHC -Wno-missing-import-lists #-}

{-# LANGUAGE OverloadedStrings #-}

import Control.Monad (replicateM_)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF hiding (placeholder, truncatedNormal, shape)
import qualified TensorFlow.Minimize as TF
import qualified TensorFlow.Ops as TF hiding (initializedVariable)
import qualified TensorFlow.Variable as TF
import qualified TensorFlow.NN as TF
import qualified Data.Vector.Storable as S
import Data.ByteString (ByteString)
import Data.Text ()
import Lens.Family2 ((.~))
import GHC.Int (Int64)

import Control.Monad.IO.Class (liftIO)

import Data.List (take)

import GenerateImages (generateImages)

main :: IO ()
main = do
    (xData, yData) <- generateImages 100 16
    fit xData yData

fit :: [[[Float]]] -> [Bool] -> IO ()
fit xData yData' = let yData = map (\x -> if x then 1.0 else 0.0) yData' in do
  vec <- TF.runSession $ do
    -- Create tensorflow constants for x and y.
    let batchSize = fromIntegral (length xData) :: Int64
        size = fromIntegral (length (head xData)) :: Int64

        flatX = concat (concat xData)

    x <- TF.placeholder (TF.Shape [batchSize, size, size])
    y <- TF.placeholder (TF.Shape [batchSize])

    -- Create scalar variables for slope and intercept.
    w1 <- TF.initializedVariable =<< TF.truncatedNormal (TF.vector [2, 2, 1, 2])
    w2 <- TF.initializedVariable =<< TF.truncatedNormal (TF.vector [2, 2, 2, 4])
    w3 <- TF.initializedVariable =<< TF.truncatedNormal (TF.vector [2, 2, 4, 4])
    w4 <- TF.initializedVariable =<< TF.truncatedNormal (TF.vector [2, 2, 4, 1])

    let vars = [ w1, w2, w3, w4 ]

    -- Define the loss function.
    let l0 = TF.reshape x (TF.vector [batchSize, size, size, 1])
        l1 = TF.conv2D' params  l0 (TF.readValue w1)
        l2 = TF.conv2D' params l1 (TF.readValue w2)
        l3 = TF.conv2D' params l2 (TF.readValue w3)
        l4 = TF.conv2D' params l3 (TF.readValue w4)
        params = (TF.opAttr "strides" .~ [1, 2, 2, 1 :: Int64])
                 . (TF.opAttr "padding" .~ ("SAME" :: ByteString))
                 . (TF.opAttr "data_format" .~ ("NHWC" :: ByteString))
                 . (TF.opAttr "use_cudnn_on_gpu" .~ True)
    -- l51 <- TF.print l5 (TF.shape l5 TF.:/ TF.Nil)
    let logits = TF.reshape l4 (TF.vector [batchSize])
        yHat = TF.sigmoid logits

    rLogits <- TF.render logits
    -- rY <- TF.render (TF.cast y)
    loss <- TF.sigmoidCrossEntropyWithLogits rLogits y
    let meanLoss = TF.reduceMean loss

    trainStep <- TF.minimizeWith (TF.gradientDescent 0.001) loss vars

    replicateM_ 1000 $ do
      () <- TF.runWithFeeds [TF.feed x (TF.encodeTensorData (TF.Shape [batchSize, size, size]) (S.fromList flatX)),
                              TF.feed y (TF.encodeTensorData (TF.Shape [batchSize]) (S.fromList yData))]
                             trainStep
      (lossVal :: S.Vector Float) <- TF.runWithFeeds [TF.feed x (TF.encodeTensorData (TF.Shape [batchSize, size, size]) (S.fromList flatX)),
                              TF.feed y (TF.encodeTensorData (TF.Shape [batchSize]) (S.fromList yData))]
                             meanLoss
      liftIO $ print ("loss" :: String)
      liftIO $ print lossVal

    (vec :: S.Vector Float) <- TF.runWithFeeds [TF.feed x (TF.encodeTensorData (TF.Shape [batchSize, size, size])  (S.fromList flatX))] yHat
    return vec

  print (take 10 yData)
  print (take 10 (S.toList vec))
  return ()
