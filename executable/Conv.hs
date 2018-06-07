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

{-# LANGUAGE PartialTypeSignatures  #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE TypeApplications     #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE RankNTypes           #-}
{-# LANGUAGE TypeOperators        #-}
{-# OPTIONS_GHC -Wno-missing-import-lists #-}


-- import GHC.TypeLits (type (*))

import Control.Monad (replicateM_)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Minimize as TF
import qualified MyOps (myConv2D)

import           Data.Vector.Sized (Vector)
import qualified Data.Vector.Sized as VS -- (replicateM, map, to)

import qualified TensorFlow.DepTyped as TFD

import Control.Monad.IO.Class (liftIO)

import Data.List (take)
import Data.Maybe (fromJust)

import GenerateImages (generateImages)

import Data.Int (Int32)

main :: IO ()
main = do
    (xData, yData) <- generateImages 100 16
    fit xData yData

fit :: [[[Float]]] -> [Bool] -> IO ()
fit xData yData' = let yData = map (\x -> if x then 1.0 else 0.0) yData' in do
  vec <- TF.runSession $ do
    -- Create tensorflow constants for x and y.
    let flatX = concat (concat xData)
        vecX :: VS.Vector (25600) Float
        vecX = fromJust $ VS.fromList flatX
        vecY :: VS.Vector 100 Float
        vecY = fromJust $ VS.fromList yData

    (x :: TFD.Placeholder "x" '[100, 16, 16] Float) <- TFD.placeholder
    (y :: TFD.Placeholder "y" '[100] Float) <- TFD.placeholder

    -- Create scalar variables for slope and intercept.
    w1 <- TFD.initializedVariable @'[2, 2, 1, 2] =<< TFD.truncatedNormal
    w2 <- TFD.initializedVariable @'[2, 2, 2, 4] =<< TFD.truncatedNormal
    w3 <- TFD.initializedVariable @'[2, 2, 4, 4] =<< TFD.truncatedNormal
    w4 <- TFD.initializedVariable @'[2, 2, 4, 1] =<< TFD.truncatedNormal

    let vars = [ TFD.unVariable w1
               , TFD.unVariable w2
               , TFD.unVariable w3
               , TFD.unVariable w4 ]

    -- Define the loss function.
    let l0 = TFD.reshape x :: TFD.Tensor '[100, 16, 16, 1] _ TFD.Build Float
        l1 = MyOps.myConv2D l0 (TFD.readValue w1) :: TFD.Tensor '[100, 8, 8, 2] _ TFD.Build Float
        l2 = MyOps.myConv2D l1 (TFD.readValue w2) :: TFD.Tensor '[100, 4, 4, 4] _ TFD.Build Float
        l3 = MyOps.myConv2D l2 (TFD.readValue w3) :: TFD.Tensor '[100, 2, 2, 4] _ TFD.Build Float
        l4 = MyOps.myConv2D l3 (TFD.readValue w4) :: TFD.Tensor '[100, 1, 1, 1] _ TFD.Build Float

    (l0shape :: Vector 4 Int32) <- TFD.runWithFeeds (TFD.feed x (TFD.encodeTensorData vecX) TFD.:~~
                                                     TFD.NilFeedList)
                                   (TFD.shape l0)
    (l1shape :: Vector 4 Int32) <- TFD.runWithFeeds (TFD.feed x (TFD.encodeTensorData vecX) TFD.:~~
                                                     TFD.NilFeedList)
                                   (TFD.shape l1)
    (l2shape :: Vector 4 Int32) <- TFD.runWithFeeds (TFD.feed x (TFD.encodeTensorData vecX) TFD.:~~
                                                     TFD.NilFeedList)
                                   (TFD.shape l2)
    (l3shape :: Vector 4 Int32) <- TFD.runWithFeeds (TFD.feed x (TFD.encodeTensorData vecX) TFD.:~~
                                                     TFD.NilFeedList)
                                   (TFD.shape l3)
    (l4shape :: Vector 4 Int32) <- TFD.runWithFeeds (TFD.feed x (TFD.encodeTensorData vecX) TFD.:~~
                                                     TFD.NilFeedList)
                                   (TFD.shape l4)
    liftIO $ print l0shape
    liftIO $ print l1shape
    liftIO $ print l2shape
    liftIO $ print l3shape
    liftIO $ print l4shape
    let logits = TFD.reshape l4
        yHat = TFD.sigmoid logits

    rLogits <- TFD.render logits
    -- rY <- TF.render (TF.cast y)
    loss <- TFD.sigmoidCrossEntropyWithLogits rLogits y
    let meanLoss = TFD.reduceMean loss

    trainStep <- TFD.minimizeWith (TF.gradientDescent 0.001) loss vars

    replicateM_ 1000 $ do
      () <- TFD.runWithFeeds ( TFD.feed x (TFD.encodeTensorData vecX) TFD.:~~
                               TFD.feed y (TFD.encodeTensorData vecY) TFD.:~~
                              TFD.NilFeedList)
                             trainStep
      (lossVal :: Vector 1 Float) <- TFD.runWithFeeds (TFD.feed x (TFD.encodeTensorData vecX) TFD.:~~
                                                       TFD.feed y (TFD.encodeTensorData vecY) TFD.:~~
                                                       TFD.NilFeedList)
                                     meanLoss
      liftIO $ print ("loss" :: String)
      liftIO $ print lossVal

    (vec :: Vector 100 Float) <- TFD.runWithFeeds (TFD.feed x (TFD.encodeTensorData vecX) TFD.:~~
                                                 TFD.NilFeedList)
                                                 yHat
    return vec

  print (take 10 yData)
  print (take 10 (VS.toList vec))
  return ()
