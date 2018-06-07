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

{-# LANGUAGE ScopedTypeVariables  #-}
{-# OPTIONS_GHC -Wno-missing-import-lists #-}

import Control.Monad (replicateM_, replicateM)
import System.Random (randomIO)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF hiding (placeholder, truncatedNormal, shape)
import qualified TensorFlow.Minimize as TF
import qualified TensorFlow.Ops as TF hiding (initializedVariable)
import qualified TensorFlow.Variable as TF
import qualified Data.Vector.Storable as S
import GHC.Int (Int32)

import Control.Monad.IO.Class (liftIO)

import Data.List (map, take)

main :: IO ()
main = do
    xData <- replicateM 1000 randomIO
    let yData = map sin xData
    fit xData yData

fit :: [Float] -> [Float] -> IO ()
fit xData yData = do
  vec <- TF.runSession $ do
    -- Create tensorflow constants for x and y.
    let
        xShape = TF.Shape [fromIntegral $ length xData]
        yShape = TF.Shape [fromIntegral $ length yData]
    x <- TF.placeholder xShape
    y <- TF.placeholder yShape
    -- Create scalar variables for slope and intercept.
    w1 <- TF.initializedVariable =<< TF.truncatedNormal (TF.vector [1, 4])
    b1 <- TF.initializedVariable =<< TF.truncatedNormal (TF.vector [4])
    w2 <- TF.initializedVariable =<< TF.truncatedNormal (TF.vector [4, 4])
    b2 <- TF.initializedVariable =<< TF.truncatedNormal (TF.vector [4])
    w3 <- TF.initializedVariable =<< TF.truncatedNormal (TF.vector [4, 1])
    b3 <- TF.initializedVariable =<< TF.truncatedNormal (TF.vector [1])

    let vars = [ w1, b1, w2, b2, w3, b3 ]

    -- Define the loss function.
    let l0 = TF.reshape x (TF.vector [-1, 1 :: Int32])
        l1 = TF.relu ((l0 `TF.matMul` TF.readValue w1) `TF.add` TF.readValue b1)
        l3 = TF.relu ((l1 `TF.matMul` TF.readValue w2) `TF.add` TF.readValue b2)
        l5 = (l3 `TF.matMul` TF.readValue w3) `TF.add` TF.readValue b3
        l6 = TF.reshape l5 (TF.vector [-1 :: Int32])
        yHat = l6 -- TF.sigmoid l6 -- sigmoid?
        loss = TF.reduceMean (TF.square (y `TF.sub` yHat))
    -- Optimize with gradient descent.


    (yHatShape:: S.Vector Int32) <- TF.runWithFeeds [TF.feed x (TF.encodeTensorData xShape (S.fromList xData)),
                              TF.feed y (TF.encodeTensorData yShape (S.fromList yData))]
                             (TF.shape yHat)
    liftIO $ print "yHatShape"
    liftIO $ print yHatShape

    trainStep <- TF.minimizeWith (TF.gradientDescent 0.00001) loss vars
    replicateM_ 1000 $ do
      () <- TF.runWithFeeds [TF.feed x (TF.encodeTensorData xShape (S.fromList xData)),
                              TF.feed y (TF.encodeTensorData yShape (S.fromList yData))]
                             trainStep
      (lossVal :: S.Vector Float) <- TF.runWithFeeds [TF.feed x (TF.encodeTensorData xShape (S.fromList xData)),
                              TF.feed y (TF.encodeTensorData yShape (S.fromList yData))]
                             loss
      liftIO $ print "loss"
      liftIO $ print lossVal

    (vec :: S.Vector Float) <- TF.runWithFeeds [TF.feed x (TF.encodeTensorData xShape  (S.fromList xData))] yHat
    return vec

  print (take 10 xData)
  print (take 10 yData)
  print (take 10 (S.toList vec))
  return ()
