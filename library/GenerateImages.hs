{-# LANGUAGE ScopedTypeVariables  #-}

module GenerateImages where

import System.Random (randomIO, randomRIO)

composeImages :: [[[Float]]] -> [Int] -> [Bool] -> [[[Float]]]
composeImages background which ydata = map
   (\(b, w, y) ->
      map (\(rowNum, row) ->
        map (\(colNum, val) ->
            if (y && colNum == w) || (not y && rowNum == w) then
              val + 1
            else val
        ) (zip [0..] row)
      ) (zip [0..] b)
   )
   (zip3 background which ydata)

generateImages :: Int -> Int -> IO ([[[Float]]], [Bool])
generateImages num size = do
  background <- mapM
      (\_ -> mapM
        (\_ -> mapM
          (\_ -> do (val :: Float) <- randomIO
                    return (val / 100.0) ) [0..size - 1]
        ) [0..size - 1]
      ) [0..num - 1]
  ydata <- mapM (\_ -> randomIO :: IO Bool) [0..num - 1]
  which <- mapM (\_ -> randomRIO (0, size - 1)) [0..num - 1]

  return (composeImages background which ydata, ydata)
