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

{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE TypeApplications     #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE RankNTypes           #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# OPTIONS_GHC -Wno-missing-import-lists #-}

import Control.Monad (replicateM_)
import System.Random (randomIO)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Minimize as TF

import Control.Monad.IO.Class (liftIO)

import           Data.Vector.Sized (Vector)
import qualified Data.Vector.Sized as VS -- (replicateM, map, to)

import qualified TensorFlow.DepTyped as TFD

import Data.List (take)

import           GHC.TypeLits (KnownNat)

main :: IO ()
main = do
    xData <- VS.replicateM @1000 randomIO
    let yData = VS.map sin xData
    -- Fit linear regression model.
    fit xData yData

fit :: forall n. KnownNat n => Vector n Float -> Vector n Float -> IO ()
fit xData yData = do
  vec <- TF.runSession $ do
    -- Create tensorflow constants for x and y.
    (x :: TFD.Placeholder "x" '[n] Float) <- TFD.placeholder
    (y :: TFD.Placeholder "y" '[n] Float) <- TFD.placeholder
    -- Create scalar variables for slope and intercept.
    w1 <- TFD.initializedVariable @'[1, 4] =<< TFD.truncatedNormal
    b1 <- TFD.initializedVariable @'[4]  =<< TFD.truncatedNormal
    w2 <- TFD.initializedVariable @'[4, 4] =<< TFD.truncatedNormal
    b2 <- TFD.initializedVariable @'[4] =<< TFD.truncatedNormal
    w3 <- TFD.initializedVariable @'[4, 1] =<< TFD.truncatedNormal
    b3 <- TFD.initializedVariable @'[1] =<< TFD.truncatedNormal

    let vars =  [ TFD.unVariable w1
                , TFD.unVariable b1
                , TFD.unVariable w2
                , TFD.unVariable b2
                , TFD.unVariable w3
                , TFD.unVariable b3 ]

    -- Define the loss function.
    let l0 = TFD.reshape x :: TFD.Tensor '[n, 1] _ _ Float
        l1 = TFD.relu ((l0 `TFD.matMul` TFD.readValue w1) `TFD.add` TFD.readValue b1)
        l3 = TFD.relu ((l1 `TFD.matMul` TFD.readValue w2) `TFD.add` TFD.readValue b2)
        l5 = (l3 `TFD.matMul` TFD.readValue w3) `TFD.add` TFD.readValue b3
        l6 = TFD.reshape l5 :: TFD.Tensor '[n] _ _ Float
        yHat = l6 -- TFD.sigmoid l6 -- sigmoid?
        loss = TFD.reduceMean (TFD.square (y `TFD.sub` yHat))
    -- Optimize with gradient descent.

    trainStep <- TFD.minimizeWith (TF.gradientDescent 0.00001) loss vars
    replicateM_ 1000 $ do
      () <- TFD.runWithFeeds (TFD.feed x (TFD.encodeTensorData xData) TFD.:~~
                              TFD.feed y (TFD.encodeTensorData yData) TFD.:~~
                              TFD.NilFeedList)
                             trainStep
      (lossVal :: Vector 1 Float) <- TFD.runWithFeeds (TFD.feed x (TFD.encodeTensorData xData) TFD.:~~
                                                     TFD.feed y (TFD.encodeTensorData yData) TFD.:~~
                                                     TFD.NilFeedList)
                                                     loss
      liftIO $ print "loss"
      liftIO $ print lossVal

    (vec :: Vector n Float) <- TFD.runWithFeeds (TFD.feed x (TFD.encodeTensorData xData) TFD.:~~
                                                 TFD.NilFeedList) yHat
    return vec

  print (take 10 (VS.toList xData))
  print (take 10 (VS.toList yData))
  print (take 10 (VS.toList vec))
  return ()
