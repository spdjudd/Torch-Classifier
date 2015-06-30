require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator

imagesize = 50
trainProportion = 0.7


-- TODO: move this to main config and default to a sensible relative folder
--sourcePath = '/home/simon/Public/Share/resized/'
sourcePath = '/home/simon/Public/Share/test/'
--testSourcePath = '/home/simon/Public/Share/test/'

-------------------------------------------------------
-- Load the data from disk

-- keep a table of classes and class sizes
classMap = {}
classFiles = {}
classSizes = {}
local popen = io.popen
for filename in popen('ls -a "'..sourcePath..'" | grep png'):lines() do
    -- files named <class>-xxxx.png
    class = string.sub(filename, 1, string.find(filename, '-') - 1)
    print(filename..' class: '..class)
    if classFiles[class] == nil then
        classFiles[class] = {}
        classSizes[class] = 0
        print('created class')
    end
    classSizes[class] = classSizes[class] + 1
    classFiles[class][classSizes[class]] = filename
end

local classCount = 0
local trainIndex = 0
local testIndex = 0
local trainTable = {}
local testTable = {}
for class,files in pairs(classFiles) do
    print('Processing class '..class)
    classCount = classCount + 1
    classMap[class] = classCount
    -- randperm the order, take the first trainProportion into train, rest into test
    -- TODO: Make sure any derived images are all in the same set (don't train on A and test on A')
    indices = torch.randperm(classSizes[class])
    trSize = trainProportion * classSizes[class]
    for i = 1,classSizes[class] do
        file = files[indices[i]]
        if i <= trSize then
            print('Train: '..file)
            trainIndex = trainIndex + 1
            trainTable[trainIndex] = file
        else
            print('Test: '..file)
            testIndex = testIndex + 1
            testTable[testIndex] = file
        end
    end
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

-- Loop over the images and load them
print('Loading training files')
for i = 1,trainIndex do
    filename = trainTable[i]
    filepath = sourcePath..filename
    class = string.sub(filename, 1, string.find(filename, '-') - 1)
    print('Path: '..filepath..' Class: '..class..' ClassIndex: '..classMap[class])
    trainData.labels[i] = classMap[class]
    trainData.data[i] = image.load(filepath,3)
end

print('Loading testing files')
for i = 1,testIndex do
    filename = testTable[i]
    filepath = sourcePath..filename
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



