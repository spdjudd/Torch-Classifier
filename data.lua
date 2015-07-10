require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator

imagesize = 50

-- TODO: move this to main config and default to a sensible relative folder
testSourcePath = '/home/simon/Public/Share/data/test/'
trainSourcePath = '/home/simon/Public/Share/data/train/'

-------------------------------------------------------
-- Load the data from disk
-- 
-- Outputs:
--  classes - table of classes the network will discriminate
--  classMap - mapping from label to class
--  trainData, trsize - tensor of training image data, plus number of images
--  testData, tesize - tensor of test image data, plus number of images

classes = {}
classMap = {}
local trainTable = {}
local trainIndex = 0
local classCount = 0

print('Building training data file list and classes...')
local popen = io.popen
for filename in popen('ls -a "'..trainSourcePath..'" | grep png'):lines() do
    -- files named <class>-xxxx.png
    class = string.sub(filename, 1, string.find(filename, '-') - 1)
    print(filename..' class: '..class)
    if classMap[class] == nil then
        classCount = classCount + 1
        classMap[class] = classCount
        classes[classCount] = class
        print('created class')
    end
    trainIndex = trainIndex + 1
    trainTable[trainIndex] = filename
end

local testTable = {}
local testIndex = 0
print('Building testing data file list...')
for filename in popen('ls -a "'..testSourcePath..'" | grep png'):lines() do
    -- files named <class>-xxxx.png
    class = string.sub(filename, 1, string.find(filename, '-') - 1)
    print(filename..' class: '..class)
    if classMap[class] == nil then
        -- throw an error - shouldn't have test classes that weren't in training data
        error('Test class <'..class..'> was not present in training data')
    end
    testIndex = testIndex + 1
    testTable[testIndex] = filename
end


trsize = trainIndex
tesize = testIndex

trainData = {
   data = torch.Tensor(trainIndex,3,50,50),
   labels = torch.Tensor(trainIndex),
   size = function() return trainIndex end
}
testData = {
   data = torch.Tensor(testIndex,3,50,50),
   labels = torch.Tensor(testIndex),
   size = function() return testIndex end
}

-- Loop over the images, in random order, and load them
indices = torch.randperm(trainIndex)
print('Loading training files')
for i = 1,trainIndex do
    filename = trainTable[indices[i]]
    filepath = trainSourcePath..filename
    class = string.sub(filename, 1, string.find(filename, '-') - 1)
    print('Path: '..filepath..' Class: '..class..' ClassIndex: '..classMap[class])
    trainData.labels[i] = classMap[class]
    trainData.data[i] = image.load(filepath,3)
end

indices = torch.randperm(testIndex)
print('Loading testing files')
for i = 1,testIndex do
    filename = testTable[indices[i]]
    filepath = testSourcePath..filename
    class = string.sub(filename, 1, string.find(filename, '-') - 1)
    print('Path: '..filepath..' Class: '..class..' ClassIndex: '..classMap[class])
    testData.labels[i] = classMap[class]
    testData.data[i] = image.load(filepath,3)
end

----------------------------------------------------------------------
print '==> preprocessing data'

-- Convert byte => float for preprocessing

trainData.data = trainData.data:float()
testData.data = testData.data:float()

-- Convert all images to YUV
print '==> preprocessing data: colorspace RGB -> YUV'
for i = 1,trainData:size() do
   trainData.data[i] = image.rgb2yuv(trainData.data[i])
end
for i = 1,testData:size() do
   testData.data[i] = image.rgb2yuv(testData.data[i])
end

-- Name channels for convenience
channels = {'y','u','v'}

-- Normalize each channel, and store mean/std
print '==> preprocessing data: normalize each feature (channel) globally'
mean = {}
std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

-- Local normalization
print '==> preprocessing data: normalize all three channels locally'

-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(13)

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- Normalize all channels locally:
for c in ipairs(channels) do
   for i = 1,trainData:size() do
      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
   end
   for i = 1,testData:size() do
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
   end
end

----------------------------------------------------------------------
print '==> verify statistics'

for i,channel in ipairs(channels) do
   trainMean = trainData.data[{ {},i }]:mean()
   trainStd = trainData.data[{ {},i }]:std()

   testMean = testData.data[{ {},i }]:mean()
   testStd = testData.data[{ {},i }]:std()

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..channel..'-channel, mean: ' .. testMean)
   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end

----------------------------------------------------------------------
print '==> visualizing data'

-- Visualization is quite easy, using itorch.image().

--if opt.visualize then
   if itorch then
   first256Samples_y = trainData.data[{ {1,256},1 }]
   first256Samples_u = trainData.data[{ {1,256},2 }]
   first256Samples_v = trainData.data[{ {1,256},3 }]
   itorch.image(first256Samples_y)
   itorch.image(first256Samples_u)
   itorch.image(first256Samples_v)
   else
      print("For visualization, run this script in an itorch notebook")
   end
--end



