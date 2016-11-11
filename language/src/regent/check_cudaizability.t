-- Copyright 2016 Stanford University
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

-- Legion Loop Cuda-izer Checker
--
-- Checks cudaizability of body of loops
--

local ast = require("regent/ast")
local data = require("common/data")
local report = require("common/report")
local std = require("regent/std")
local symbol_table = require("regent/symbol_table")

local min = math.min

local bounds_checks = std.config["bounds-checks"]

-- cudaizer

local SIMD_REG_SIZE
if os.execute("bash -c \"[ `uname` == 'Darwin' ]\"") == 0 then
  if os.execute("sysctl -a | grep machdep.cpu.features | grep AVX > /dev/null") == 0 then
    SIMD_REG_SIZE = 32
  elseif os.execute("sysctl -a | grep machdep.cpu.features | grep SSE > /dev/null") == 0 then
    SIMD_REG_SIZE = 16
  else
    error("Unable to determine CPU architecture")
  end
else
  if os.execute("grep avx /proc/cpuinfo > /dev/null") == 0 then
    SIMD_REG_SIZE = 32
  elseif os.execute("grep sse /proc/cpuinfo > /dev/null") == 0 then
    SIMD_REG_SIZE = 16
  else
    error("Unable to determine CPU architecture")
  end
end

local reduction = nil

local V = {}
V.__index = V
function V:__tostring() return "vector" end
setmetatable(V, V)
local S = {}
S.__index = S
function S:__tostring() return "scalar" end
setmetatable(S, S)
local C = {}
C.__index = C
function C:__tostring() return "vector-contiguous" end
setmetatable(C, C)

local function join(fact1, fact2)
  if fact1 == S then return fact2
  elseif fact2 == S then return fact1
  elseif fact1 == V or fact2 == V then return V
  else assert(fact1 == C and fact2 == C) return C end
end

local context = {}
context.__index = context

function context:new_local_scope()
  local cx = {
    var_type = self.var_type:new_local_scope(),
    subst = self.subst:new_local_scope(),
    expr_type = self.expr_type,
    demanded = self.demanded,
    read_set = self.read_set,
    loop_symbol = self.loop_symbol,
  }
  return setmetatable(cx, context)
end

function context:new_global_scope(loop_symbol)
  local cx = {
    var_type = symbol_table:new_global_scope(),
    subst = symbol_table:new_global_scope(),
    expr_type = {},
    demanded = false,
    read_set = {},
    write_set = {},
    loop_symbol = loop_symbol,
  }
  return setmetatable(cx, context)
end

function context:assign(symbol, fact)
  self.var_type:insert(nil, symbol, fact)
end

function context:join(symbol, fact)
  local var_type = self.var_type
  local old_fact = var_type:safe_lookup(symbol)
  assert(old_fact)
  local new_fact = join(old_fact, fact)
  var_type:insert(nil, symbol, new_fact)
end

function context:lookup_expr_type(node)
  return self.expr_type[node]
end

function context:assign_expr_type(node, fact)
  self.expr_type[node] = fact
end

function context:join_expr_type(node, fact)
  self.expr_type[node] = join(fact, self.expr_type[node])
end

function context:report_error_when_demanded(node, error_msg)
  if self.demanded then report.error(node, error_msg) end
end

function context:add_substitution(from, to)
  self.subst:insert(nil, from, to)
end

function context:find_replacement(from)
  local to = self.subst:safe_lookup(from)
  assert(to)
  return to
end

local function get_bounds(ty)
  if std.is_ref(ty) then
    return ty:bounds():map(function(bound)
      return data.newtuple(bound, ty.field_path)
    end)
  else
    return terralib.newlist()
  end
end

local function collect_bounds(node)
  local bounds = terralib.newlist()
  if node:is(ast.typed.expr.FieldAccess) or
     node:is(ast.typed.expr.Deref) then
    local ty = node.expr_type
    if std.is_ref(ty) then bounds:insertall(get_bounds(ty)) end
    bounds:insertall(collect_bounds(node.value))

  elseif node:is(ast.typed.expr.IndexAccess) then
    bounds:insertall(collect_bounds(node.value))
    bounds:insertall(collect_bounds(node.index))

  elseif node:is(ast.typed.expr.Unary) then
    bounds:insertall(collect_bounds(node.rhs))

  elseif node:is(ast.typed.expr.Binary) then
    bounds:insertall(collect_bounds(node.lhs))
    bounds:insertall(collect_bounds(node.rhs))

  elseif node:is(ast.typed.expr.Ctor) then
    for _, field in pairs(node.fields) do
      bounds:insertall(collect_bounds(field))
    end

  elseif node:is(ast.typed.expr.CtorRecField) then
    bounds:insertall(collect_bounds(node.value))

  elseif node:is(ast.typed.expr.Call) then
    for _, arg in pairs(node.args) do
      bounds:insertall(collect_bounds(arg))
    end

  elseif node:is(ast.typed.expr.Cast) then
    bounds:insertall(collect_bounds(node.arg))
  end

  return bounds
end

-- cudaizability check returns truen when the statement is cudaizable
local check_cudaizability = {}
local error_prefix = "cudaization failed: loop body has "

function check_cudaizability.get_reduction_node()
  return reduction
end

function check_cudaizability.clear_reduction_node()
  reduction = nil
end

check_cudaizability.pass_name = "check_cudaizability"

function check_cudaizability.entry(node)
  if node:is(ast.typed.top.Task) then
    local cudaizable = true
    node.body.stats:map(function (stat)
      if stat:is(ast.typed.stat.ForList) then
        local cx = context:new_global_scope(stat.symbol)
        cx:assign(stat.symbol, C)
        cx.demanded = true
        reduction = nil

        cudaizable = check_cudaizability.block(cx, stat.block)
      end
    end)
    if not cudaizable then error("code not CUDAizable") end
  end
  return node
end

function check_cudaizability.block(cx, node)
  cx = cx:new_local_scope()
  for i, stat in ipairs(node.stats) do
    local cudaizable = check_cudaizability.stat(cx, stat)
    if not cudaizable then return false end
  end
  return true
end

-- reject an aliasing between the read set and write set
function check_cudaizability.check_aliasing(cx, write_set)
  -- ignore write accesses directly to the region being iterated over
  cx.loop_symbol:gettype():bounds():map(function(r)
    write_set[r] = nil
  end)
  for ty, fields in pairs(write_set) do
    if cx.read_set[ty] then
      for field_hash, pair in pairs(fields) do
        if cx.read_set[ty][field_hash] then
          local field, node = unpack(pair)
          local path = ""
          if #field > 0 then
            path = "." .. field[1]
            for idx = 2, #field do
              path = path .. "." .. field[idx]
            end
          end
          cx:report_error_when_demanded(node, error_prefix ..
            "aliasing update of path " ..  tostring(ty) .. path)
          return false
        end
      end
    end
  end
end

function check_cudaizability.stat(cx, node)
  if node:is(ast.typed.stat.Block) then
    return check_cudaizability.block(cx, node.block)

  elseif node:is(ast.typed.stat.Var) then
    for i, symbol in pairs(node.symbols) do
      local fact = V
      if #node.values > 0 then
        local value = node.values[i]
        if not check_cudaizability.expr(cx, value) then return false end
        fact = cx:lookup_expr_type(value)
      end

      cx:assign(symbol, fact)
      node.values:map(function(value)
        collect_bounds(value):map(function(pair)
          local ty, field = unpack(pair)
          local field_hash = field:hash()
          if not cx.read_set[ty] then cx.read_set[ty] = {} end
          if not cx.read_set[ty][field_hash] then
            cx.read_set[ty][field_hash] = data.newtuple(field, node)
          end
        end)
      end)
    end
    return true

  elseif node:is(ast.typed.stat.Assignment) or
         node:is(ast.typed.stat.Reduce) then
    for i, rh in pairs(node.rhs) do
      local lh = node.lhs[i]

      if not check_cudaizability.expr(cx, lh) or
         not check_cudaizability.expr(cx, rh) then return false end

      if node:is(ast.typed.stat.Reduce) and cx:lookup_expr_type(lh) == S then
        if reduction then
          cx:report_error_when_demanded(node, error_prefix ..
            "multiple reduction statements")
          return false
        end

        reduction = node

        if node.op ~= "+" and node.op ~= "*" and node.op ~= "max" and node.op ~= "min" then
            cx:report_error_when_demanded(node, error_prefix ..
              "unsupported reduction operation")
            return false
        end
      else
          if cx:lookup_expr_type(lh) == S and cx:lookup_expr_type(rh) == V then
            cx:report_error_when_demanded(node, error_prefix ..
              "an assignment of a non-scalar expression to a scalar expression")
            return false
          end

          -- TODO: we could accept statements with no loop carrying dependence
          if cx:lookup_expr_type(lh) == S then
            cx:report_error_when_demanded(node, error_prefix ..
              "an assignment to a scalar expression")
            return false
          end
      end

      -- TODO: for the moment we reject an assignment such as
      -- 'r[i] = i' where 'i' is of an index type
      if std.is_bounded_type(rh.expr_type) and
         rh.expr_type.dim >= 1 then
        cx:report_error_when_demanded(node, error_prefix ..
          "a corner case statement not supported for the moment")
        return false
      end

      -- bookkeeping for alias analysis
      collect_bounds(rh):map(function(pair)
        local ty, field = unpack(pair)
        local field_hash = field:hash()
        if not cx.read_set[ty] then cx.read_set[ty] = {} end
        if not cx.read_set[ty][field_hash] then
          cx.read_set[ty][field_hash] = data.newtuple(field, node)
        end
      end)
      local write_set = {}
      get_bounds(lh.expr_type):map(function(pair)
        local ty, field = unpack(pair)
        local field_hash = field:hash()
        if not write_set[ty] then write_set[ty] = {} end
        if not write_set[ty][field_hash] then
          write_set[ty][field_hash] = data.newtuple(field, node)
        end
      end)
      check_cudaizability.check_aliasing(cx, write_set)
    end

    return true

  elseif node:is(ast.typed.stat.ForNum) then
    for _, value in pairs(node.values) do
      if not check_cudaizability.expr(cx, value) then return false end
      if cx:lookup_expr_type(value) ~= S then
        cx:report_error_when_demanded(node,
          error_prefix ..  "a non-scalar loop condition")
        return false
      end
    end
    cx = cx:new_local_scope()
    cx:assign(node.symbol, S)
    -- check loop body twice to check loop carried aliasing
    for idx = 0, 2 do
      if not check_cudaizability.block(cx, node.block) then
        return false
      end
    end
    return true

  elseif node:is(ast.typed.stat.If) then
    -- bookkeeping for alias analysis
    collect_bounds(node.cond):map(function(pair)
      local ty, field = unpack(pair)
      local field_hash = field:hash()
      if not cx.read_set[ty] then cx.read_set[ty] = {} end
      if not cx.read_set[ty][field_hash] then
        cx.read_set[ty][field_hash] = data.newtuple(field, node)
      end
    end)

    if not check_cudaizability.expr(cx, node.cond) then return false end

    if not check_cudaizability.block(cx, node.then_block) then
      return false
    end

    for _, elseif_block in ipairs(node.elseif_blocks) do
      if not check_cudaizability.stat(cx, elseif_block) then return false end
    end

    return check_cudaizability.block(cx, node.else_block)

  elseif node:is(ast.typed.stat.Elseif) then
    if not check_cudaizability.expr(cx, node.cond) then return false end
    if cx:lookup_expr_type(node.cond) ~= S then
      cx:report_error_when_demanded(node,
        error_prefix ..  "a non-scalar if-condition")
      return false
    end

    return check_cudaizability.block(cx, node.block)

  else
    if node:is(ast.typed.stat.While) then
      cx:report_error_when_demanded(node, error_prefix .. "an inner loop")

    elseif node:is(ast.typed.stat.ForList) then
      cx:report_error_when_demanded(node, error_prefix .. "an inner loop")

    elseif node:is(ast.typed.stat.Repeat) then
      cx:report_error_when_demanded(node, error_prefix .. "an inner loop")

    elseif node:is(ast.typed.stat.VarUnpack) then
      cx:report_error_when_demanded(node, error_prefix .. "an unpack statement")

    elseif node:is(ast.typed.stat.Return) then
      cx:report_error_when_demanded(node, error_prefix .. "a return statement")

    elseif node:is(ast.typed.stat.Break) then
      cx:report_error_when_demanded(node, error_prefix .. "a break statement")

    elseif node:is(ast.typed.stat.Expr) then
      cx:report_error_when_demanded(node,
        error_prefix .. "an expression as a statement")

    elseif node:is(ast.typed.stat.BeginTrace) then
      cx:report_error_when_demanded(node, error_prefix .. "a trace statement")

    elseif node:is(ast.typed.stat.EndTrace) then
      cx:report_error_when_demanded(node, error_prefix .. "a trace statement")

    else
      assert(false, "unexpected node type " .. tostring(node:type()))
    end

    return false
  end
end

function check_cudaizability.expr(cx, node)
  if node:is(ast.typed.expr.ID) then
    -- treats variables from the outer scope as scalars
    local fact = cx.var_type:safe_lookup(node.value) or S
    cx:assign_expr_type(node, fact)
    return true

  elseif node:is(ast.typed.expr.FieldAccess) then
    if not check_cudaizability.expr(cx, node.value) then return false end
    if cx:lookup_expr_type(node.value) == C and
       not std.is_ref(node.expr_type) then
      cx:report_error_when_demanded(node, error_prefix ..
        "an access to loop indicies")
      return false
    end
    cx:assign_expr_type(node, cx:lookup_expr_type(node.value))
    return true

  elseif node:is(ast.typed.expr.IndexAccess) then
    if not check_cudaizability.expr(cx, node.value) or
       not check_cudaizability.expr(cx, node.index) then
      return false
    end

    local fact =
      join(cx:lookup_expr_type(node.value), cx:lookup_expr_type(node.index))
    if fact == C then fact = V end
    cx:assign_expr_type(node, fact)
    return true

  elseif node:is(ast.typed.expr.Unary) then
    if not check_cudaizability.expr(cx, node.rhs) then return true end
    cx:assign_expr_type(node, cx:lookup_expr_type(node.rhs))
    return true

  elseif node:is(ast.typed.expr.Binary) then
    if not check_cudaizability.binary_op(node.op,
                                           std.as_read(node.expr_type)) then
      cx:report_error_when_demanded(node,
        error_prefix .. "an unsupported binary operator")
      return false
    end

    if not check_cudaizability.expr(cx, node.lhs) or
       not check_cudaizability.expr(cx, node.rhs) then
      return false
    end

    local fact =
      join(cx:lookup_expr_type(node.lhs), cx:lookup_expr_type(node.rhs))
    if std.is_index_type(std.as_read(node.expr_type)) and
       node.op == "%" and fact ~= S then
       fact = V
    end
    cx:assign_expr_type(node, fact)
    return true

  elseif node:is(ast.typed.expr.Ctor) then
    cx:assign_expr_type(node, S)
    for _, field in pairs(node.fields) do
      if not check_cudaizability.expr(cx, field) then return false end
      cx:join_expr_type(node, cx:lookup_expr_type(field))
    end
    return true

  elseif node:is(ast.typed.expr.CtorRecField) or
         node:is(ast.typed.expr.CtorListField) then
    if not check_cudaizability.expr(cx, node.value) then return false end
    cx:assign_expr_type(node, cx:lookup_expr_type(node.value))
    return true

  elseif node:is(ast.typed.expr.Constant) then
    cx:assign_expr_type(node, S)
    return true

  elseif node:is(ast.typed.expr.Call) then
    return check_cudaizability.expr_call(cx, node)

  elseif node:is(ast.typed.expr.Cast) then
    if not check_cudaizability.expr(cx, node.arg) then return false end
    if std.is_bounded_type(node.arg.expr_type) and
       node.arg.expr_type.dim >= 1 then
      cx:report_error_when_demanded(node, error_prefix ..
        "a corner case statement not supported for the moment")
      return false
    end
    cx:assign_expr_type(node, cx:lookup_expr_type(node.arg))
    return true

  elseif node:is(ast.typed.expr.UnsafeCast) then
    if not check_cudaizability.expr(cx, node.value) then return false end
    cx:assign_expr_type(node, cx:lookup_expr_type(node.value))
    return true

  elseif node:is(ast.typed.expr.Deref) then
    if not check_cudaizability.expr(cx, node.value) then return false end
    local fact = cx:lookup_expr_type(node.value)
    if fact == C then fact = V end
    cx:assign_expr_type(node, fact)
    return true

  else
    if node:is(ast.typed.expr.MethodCall) then
      cx:report_error_when_demanded(node, error_prefix .. "a method call")

    elseif node:is(ast.typed.expr.RawContext) then
      cx:report_error_when_demanded(node, error_prefix .. "a raw expression")

    elseif node:is(ast.typed.expr.RawFields) then
      cx:report_error_when_demanded(node, error_prefix .. "a raw expression")

    elseif node:is(ast.typed.expr.RawPhysical) then
      cx:report_error_when_demanded(node, error_prefix .. "a raw expression")

    elseif node:is(ast.typed.expr.RawRuntime) then
      cx:report_error_when_demanded(node, error_prefix .. "a raw expression")

    elseif node:is(ast.typed.expr.Isnull) then
      cx:report_error_when_demanded(node,
        error_prefix .. "an isnull expression")

    elseif node:is(ast.typed.expr.New) then
      cx:report_error_when_demanded(node, error_prefix .. "a new expression")

    elseif node:is(ast.typed.expr.Null) then
      cx:report_error_when_demanded(node, error_prefix .. "a null expression")

    elseif node:is(ast.typed.expr.DynamicCast) then
      cx:report_error_when_demanded(node, error_prefix .. "a dynamic cast")

    elseif node:is(ast.typed.expr.StaticCast) then
      cx:report_error_when_demanded(node, error_prefix .. "a static cast")

    elseif node:is(ast.typed.expr.Region) then
      cx:report_error_when_demanded(node, error_prefix .. "a region expression")

    elseif node:is(ast.typed.expr.Partition) then
      cx:report_error_when_demanded(node,
        error_prefix .. "a patition expression")

    elseif node:is(ast.typed.expr.CrossProduct) then
      cx:report_error_when_demanded(node,
        error_prefix .. "a cross product operation")

    elseif node:is(ast.typed.expr.Function) then
      cx:report_error_when_demanded(node,
        error_prefix .. "a function reference")

    elseif node:is(ast.typed.expr.RawValue) then
      cx:report_error_when_demanded(node,
        error_prefix .. "a raw operator")

    else
      assert(false, "unexpected node type " .. tostring(node:type()))
    end

    return false
  end
end

local predefined_functions = {}

function check_cudaizability.expr_call(cx, node)
  assert(node:is(ast.typed.expr.Call))

  if std.is_math_op(node.fn.value) then
    local fact = S
    for _, arg in pairs(node.args) do
      if not check_cudaizability.expr(cx, arg) then return false end
      fact = join(fact, cx:lookup_expr_type(arg))
    end
    cx:assign_expr_type(node, fact)
    return true

  else
    cx:report_error_when_demanded(node,
      error_prefix .. "an unsupported function call")
    return false
  end
end

function check_cudaizability.binary_op(op, arg_type)
  if (op == "max" or op == "min") and
     not (arg_type == float or arg_type == double) then
    return false
  end
  return arg_type:isprimitive() or
         std.is_index_type(arg_type)
end

return check_cudaizability
