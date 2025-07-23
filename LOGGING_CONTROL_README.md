# 🎛️ Real-Steel Logging Control System

## ✅ Problem Solved

The excessive logging output has been resolved with a flexible logging control system that lets you choose your preferred level of detail.

## 🔊 Available Logging Levels

### 1. 🔇 **Silent Mode**
- **Minimal output** for production use
- Only essential system messages
- Clean execution with minimal console noise
- **Best for**: Production runs, clean recordings

```bash
python real_steel_launcher.py --demo --log-level silent
```

### 2. 📢 **Normal Mode** (Default)
- **Standard informational messages**
- System status and important events
- Error messages and warnings
- **Best for**: Regular use, development

```bash
python real_steel_launcher.py --demo --log-level normal
```

### 3. 🔊 **Verbose Mode**
- **Detailed method call tracking**
- Shows IK solver function calls
- Optimization progress indicators
- **Best for**: Debugging IK issues, performance analysis

```bash
python real_steel_launcher.py --demo --log-level verbose
```

### 4. 🐛 **Debug Mode**
- **Full debug information**
- Complete method call tracing
- Timing information
- Internal state logging
- **Best for**: Deep debugging, development

```bash
python real_steel_launcher.py --demo --log-level debug
```

## 🚀 How to Use

### Interactive Launcher
```bash
python real_steel_launcher.py
```
- Presents a menu to select execution mode
- Choose logging level interactively
- User-friendly interface with emojis

### Direct Command Line
```bash
# Run demo with silent logging
python real_steel_launcher.py --demo --log-level silent

# Run live mode with verbose logging
python real_steel_launcher.py --live --log-level verbose

# Run tests with debug logging
python real_steel_launcher.py --test --log-level debug
```

### Original Methods Still Work
```bash
# Original methods still functional
python src/core/main.py --ik-backend analytical
python demo_enhanced_features.py
```

## 🎯 Execution Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| 📊 **Demo** | Feature demonstration | Show all capabilities |
| 🎮 **Live** | Real-time retargeting | Actual motion capture |
| 🧪 **Test** | System validation | Verify functionality |
| 📖 **Help** | Detailed documentation | Learn system features |

## 🔧 Technical Details

### Logging Control Implementation
- **Global flags**: `DEBUG_METHOD_CALLS` in each module
- **Function decorators**: `@log_method_call()` for selective logging
- **Runtime control**: Functions to enable/disable logging
- **Module-specific**: Separate control for analytical vs BRPSO

### Output Examples

**Silent Mode:**
```
🔇 SILENT MODE: Minimal logging enabled
✅ System initialized with 14 joint angles
🎉 All tests passed!
```

**Verbose Mode:**
```
🔊 VERBOSE MODE: Detailed method call logging enabled
[IKAlternative] -> solve()
[IKAlternative] -> initialize_swarm()
[IKAlternative] -> objective_function()
[IKAlternative] -> forward_kinematics()
```

## 💡 Recommendations

- **For daily use**: Normal mode (default)
- **For demos/presentations**: Silent mode
- **For debugging IK issues**: Verbose mode
- **For deep troubleshooting**: Debug mode

## 🔄 Runtime Switching

Even during execution, you can control logging by pressing keys:
- Press `V` to toggle visualizations
- Press `I` to switch IK solvers
- Each mode respects your chosen logging level

---

**🎉 Your Real-Steel system now has clean, controllable output!** 