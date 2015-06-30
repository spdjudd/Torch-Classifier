----------------------------------------------------------------------------------
-- Basic web service using turbo, to identify images using a pre-trained network
--
----------------------------------------------------------------------------------

require('mobdebug').start()
local turbo = require("turbo")
local torch = require("torch")
local image = require("image")
local nn = require("nn")
local socket = require("socket")

-- Load the pre-trained network
local m = torch.load('model.net')
-- set it to evaluate mode
m:evaluate()
print("loaded network")
local channels = {'y','u','v'}
local classMap = {
  "longtailedtit",
  "coaltit",
  "wren",
  "chaffinch",
  "blackbirdf",
  "nuthatch",
  "greenwoodpecker",
  "magpie",
  "collareddove",
  "robin",
  "blackbirdm",
  "woodpigeon",
  "starling",
  "thrush",
  "dunnock",
  "greaterspottedwoodpecker",
  "bluetit",
  "goldfinch",
  "greenfinch",
  "greattit"
}

-- normalisation values from network training set
local mean = { 0.48040727135325, -0.046775947718871, 0.010413669464772 }
local std = {  0.20588362637994,  0.056111722980383,  0.052994471923805 }
local neighborhood = image.gaussian1D(13)
local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

local function printf(...)
    io.write(string.format(...))
end

-- encoding
local b='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

function enc(data)
    return ((data:gsub('.', function(x) 
        local r,b='',x:byte()
        for i=8,1,-1 do r=r..(b%2^i-b%2^(i-1)>0 and '1' or '0') end
        return r;
    end)..'0000'):gsub('%d%d%d?%d?%d?%d?', function(x)
        if (#x < 6) then return '' end
        local c=0
        for i=1,6 do c=c+(x:sub(i,i)=='1' and 2^(6-i) or 0) end
        return b:sub(c+1,c+1)
    end)..({ '', '==', '=' })[#data%3+1])
end

-- decoding
function dec(data)
    data = string.gsub(data, '[^'..b..'=]', '')
    return (data:gsub('.', function(x)
        if (x == '=') then return '' end
        local r,f='',(b:find(x)-1)
        for i=6,1,-1 do r=r..(f%2^i-f%2^(i-1)>0 and '1' or '0') end
        return r;
    end):gsub('%d%d%d?%d?%d?%d?%d?%d?', function(x)
        if (#x ~= 8) then return '' end
        local c=0
        for i=1,8 do c=c+(x:sub(i,i)=='1' and 2^(8-i) or 0) end
        return string.char(c)
    end))
end

-- Test requesthandler with a method get() for HTTP GET.
local TestHandler = class("TestHandler", turbo.web.RequestHandler)
function TestHandler:get()
    msg = "Test message!"
    self:write({message=msg})
end

-- Main handler for image identification, for HTTP POST
local IdBirdHandler = class("IdBirdHandler", turbo.web.RequestHandler)
function IdBirdHandler:post()
    local t0 = socket.gettime()
    local imgData = self:get_argument("img")

    -- decode the base64
    local imgBytes = turbo.escape.base64_decode(imgData)
    local byteCount = string.len(imgBytes)

    -- put into a torch tensor
    local bStorage = torch.ByteStorage():string(imgBytes)
    local dStorage = torch.Storage(byteCount):copy(bStorage)
    local ocvTensor = torch.Tensor(dStorage, 1, torch.LongStorage{50,52,3})
    local imgTensor = torch.Tensor(3, 50, 50)
    for c = 1,3 do
        imgTensor[c] = ocvTensor[{{},{1,50},c}]
    end
    imgTensor:div(255)

    -- change to yuv, 
    imgTensor = imgTensor:float()
    imgTensor = image.rgb2yuv(imgTensor)

    -- normalize each channel globally (do we need this?)
    for i,channel in ipairs(channels) do
       imgTensor[{ i,{},{} }]:add(-mean[i])
       imgTensor[{ i,{},{} }]:div(std[i])
    end

    -- normalize locally
    for c in ipairs(channels) do
        imgTensor[{ {c},{},{} }] = normalization:forward(imgTensor[{ {c},{},{} }])
    end

    -- send to nn and get response
    local output = m:forward(imgTensor:double())
    result = {}
    for i = 1,20 do
        result[i] = {result=classMap[i], p=output[i]}
    end
    self:write(result)
    print(string.format("Duration: %d ms", (socket.gettime() - t0) * 1000))
end

-- Create an Application object and bind our HelloWorldHandler to the route '/hello'.
local app = turbo.web.Application:new({
    {"/test", TestHandler},
    {"/idbird", IdBirdHandler}
})

-- Set the server to listen on port 8888 and start the ioloop.
_G.TURBO_SOCKET_BUFFER_SZ = 200000 
app:listen(8888)
turbo.ioloop.instance():start()
