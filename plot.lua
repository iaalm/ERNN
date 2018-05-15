--local name = 'workdir_base_2.live.10006.cell'
local name = arg[1]:gsub('/$',''):gsub('/','.')..".cell"
local cell = require(name)
local g = cell.cell(2,4,3)
graph.dot(g.fg, name, arg[1]..'/cell')
graph.dot(g.fg, name, 'gen/'..name)
