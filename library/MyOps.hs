{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE RankNTypes           #-}
{-# LANGUAGE KindSignatures       #-}
{-# LANGUAGE TypeOperators        #-}
{-# LANGUAGE OverloadedStrings    #-}
{-# LANGUAGE TypeFamilies         #-}
{-# OPTIONS_GHC -Wno-missing-import-lists #-}

module MyOps where

import           Data.Kind (Type)
import           GHC.TypeLits (KnownNat, Nat, Symbol, KnownNat, type (+) )

import TensorFlow.DepTyped
import  TensorFlow.DepTyped.Tensor (Tensor (Tensor))

import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF (conv2D')
import           TensorFlow.DepTyped.Base (UnionPlaceholder)
import qualified TensorFlow.NN as TF (sigmoidCrossEntropyWithLogits)
import Lens.Family2 ((.~))
import Data.Text ()

import           Data.Vector.Sized ()
import           Data.Int (Int64)
import           Data.Word (Word16)
import           Data.ByteString (ByteString)

myConv2D :: forall bs w h cIn cOut v'1 v'2 a phs1 phs2.
         (KnownNat bs, KnownNat w, KnownNat h, TF.OneOf '[Word16, Float] a)
         => Tensor '[bs, w + w, h + h, cIn] phs1 v'1 a
         -> Tensor '[2, 2, cIn, cOut] phs2 v'2 a
         -> Tensor '[bs, w, h, cOut] (UnionPlaceholder phs1 phs2) Build a
myConv2D (Tensor input) (Tensor fil) = Tensor (TF.conv2D' params input fil)
  where
    params = (TF.opAttr "strides" .~ [1, 2, 2, 1 :: Int64])
             . (TF.opAttr "padding" .~ ("SAME" :: ByteString))
             . (TF.opAttr "data_format" .~ ("NHWC" :: ByteString))
             . (TF.opAttr "use_cudnn_on_gpu" .~ True)
