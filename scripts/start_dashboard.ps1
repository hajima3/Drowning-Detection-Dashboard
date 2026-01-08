# YOLOv11 Drowning Detection Dashboard Launcher
# PowerShell script for Windows

# Color output functions
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-ColorOutput Cyan "=========================================="
Write-ColorOutput Cyan "  YOLOv11 Drowning Detection Dashboard"
Write-ColorOutput Cyan "=========================================="
Write-Output ""

# Check if Python is installed
Write-ColorOutput Yellow "Checking Python installation..."
try {
    $pythonVersion = python --version 2>&1
    Write-ColorOutput Green "✓ $pythonVersion"
} catch {
    Write-ColorOutput Red "✗ Python not found!"
    Write-ColorOutput Red "Please install Python 3.8+ from https://www.python.org/downloads/"
    Write-ColorOutput Red "Make sure to check 'Add Python to PATH' during installation."
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Output ""

# Check if virtual environment exists
Write-ColorOutput Yellow "Checking virtual environment..."
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-ColorOutput Green "✓ Virtual environment found"
} else {
    Write-ColorOutput Yellow "Virtual environment not found. Creating..."
    python -m venv .venv
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "✓ Virtual environment created"
    } else {
        Write-ColorOutput Red "✗ Failed to create virtual environment"
        Read-Host "Press Enter to exit"
        exit 1
    }
}
Write-Output ""

# Activate virtual environment
Write-ColorOutput Yellow "Activating virtual environment..."
try {
    & .\.venv\Scripts\Activate.ps1
    Write-ColorOutput Green "✓ Virtual environment activated"
} catch {
    Write-ColorOutput Red "✗ Failed to activate virtual environment"
    Write-ColorOutput Yellow "If you see 'execution policy' error, run:"
    Write-ColorOutput Yellow "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Output ""

# Check if dependencies are installed
Write-ColorOutput Yellow "Checking dependencies..."
$packagesInstalled = $true
$requiredPackages = @("flask", "ultralytics", "opencv-python", "torch")

foreach ($package in $requiredPackages) {
    $installed = pip show $package 2>&1 | Select-String "Name:"
    if (-not $installed) {
        $packagesInstalled = $false
        Write-ColorOutput Yellow "✗ Package '$package' not found"
        break
    }
}

if (-not $packagesInstalled) {
    Write-ColorOutput Yellow "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "✓ Dependencies installed successfully"
    } else {
        Write-ColorOutput Red "✗ Failed to install dependencies"
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-ColorOutput Green "✓ All dependencies installed"
}
Write-Output ""

# Check if model file exists
Write-ColorOutput Yellow "Checking model file..."
if (Test-Path ".\best.pt") {
    Write-ColorOutput Green "✓ Model file found (best.pt)"
} else {
    Write-ColorOutput Red "✗ Model file 'best.pt' not found!"
    Write-ColorOutput Yellow "Please download or train the model file and place it in the project root."
    Write-ColorOutput Yellow "See MODEL_README.md for instructions."
    Write-Output ""
    $continue = Read-Host "Continue anyway? (will use base YOLO model) [y/N]"
    if ($continue -ne "y" -and $continue -ne "Y") {
        exit 1
    }
}
Write-Output ""

# Start the Flask application
Write-ColorOutput Cyan "=========================================="
Write-ColorOutput Cyan "  Starting Dashboard Server"
Write-ColorOutput Cyan "=========================================="
Write-Output ""
Write-ColorOutput Green "Dashboard will be available at:"
Write-ColorOutput Green "  - http://localhost:5000"
Write-ColorOutput Green "  - http://127.0.0.1:5000"
Write-Output ""

# Get local IP address
$ipAddress = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.InterfaceAlias -notlike "*Loopback*" -and $_.IPAddress -notlike "169.254.*"} | Select-Object -First 1).IPAddress
if ($ipAddress) {
    Write-ColorOutput Green "  - http://${ipAddress}:5000 (LAN access)"
    Write-Output ""
}

Write-ColorOutput Yellow "Press Ctrl+C to stop the server"
Write-Output ""
Write-ColorOutput Cyan "=========================================="
Write-Output ""

# Run the Flask application
try {
    python app.py
} catch {
    Write-ColorOutput Red "✗ Failed to start application"
    Write-ColorOutput Red "Error: $_"
    Read-Host "Press Enter to exit"
    exit 1
}

# Cleanup (runs when Ctrl+C is pressed)
Write-Output ""
Write-ColorOutput Yellow "Shutting down..."
Write-ColorOutput Green "Dashboard stopped successfully"
Write-Output ""
Read-Host "Press Enter to exit"
