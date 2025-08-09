# Stock Prediction Hub - Auto Updater PowerShell Script
# Runs every 15 minutes to update market data and commit to Git

param(
    [string]$Action = "menu",
    [string]$ProjectPath = $PWD.Path
)

# Set console colors and title
$Host.UI.RawUI.WindowTitle = "Stock Prediction Hub - Auto Updater"
$Host.UI.RawUI.BackgroundColor = "DarkBlue"
$Host.UI.RawUI.ForegroundColor = "White"
Clear-Host

function Show-Banner {
    Write-Host "===========================================" -ForegroundColor Cyan
    Write-Host "   🚀 Stock Prediction Hub Auto-Updater   " -ForegroundColor Yellow
    Write-Host "===========================================" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Menu {
    Show-Banner
    Write-Host "📋 Auto-Update System Options:" -ForegroundColor Green
    Write-Host ""
    Write-Host "[1] 🚀 Start Auto-Updater (Every 15 minutes)" -ForegroundColor White
    Write-Host "[2] 🔄 Run Single Update Now" -ForegroundColor White
    Write-Host "[3] 🛑 Stop Auto-Updater" -ForegroundColor White
    Write-Host "[4] 📋 View Update Log" -ForegroundColor White
    Write-Host "[5] 📊 Check System Status" -ForegroundColor White
    Write-Host "[6] ⚙️  Install Dependencies" -ForegroundColor White
    Write-Host "[7] 🔧 Setup Git Configuration" -ForegroundColor White
    Write-Host "[8] 🚪 Exit" -ForegroundColor White
    Write-Host ""
}

function Start-AutoUpdater {
    Write-Host "🚀 Starting Auto-Updater..." -ForegroundColor Green
    Write-Host "📅 Schedule: Every 15 minutes" -ForegroundColor Yellow
    Write-Host "🎯 This will update market data and commit changes to Git" -ForegroundColor Yellow
    Write-Host "⚠️  Press Ctrl+C to stop the auto-updater" -ForegroundColor Red
    Write-Host ""
    
    try {
        & python auto_updater.py --path $ProjectPath
    }
    catch {
        Write-Host "❌ Error starting auto-updater: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "🔧 Make sure Python and dependencies are installed" -ForegroundColor Yellow
    }
}

function Invoke-SingleUpdate {
    Write-Host "🔄 Running single update cycle..." -ForegroundColor Green
    Write-Host ""
    
    try {
        & python auto_updater.py --path $ProjectPath --once
        Write-Host ""
        Write-Host "✅ Single update completed!" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Error running update: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Write-Host ""
    Read-Host "Press Enter to continue"
}

function Stop-AutoUpdater {
    Write-Host "🛑 Stopping Auto-Updater processes..." -ForegroundColor Yellow
    
    try {
        Get-Process | Where-Object { $_.ProcessName -eq "python" -and $_.MainWindowTitle -like "*auto_updater*" } | Stop-Process -Force
        Write-Host "✅ Auto-updater processes stopped" -ForegroundColor Green
    }
    catch {
        Write-Host "ℹ️  No auto-updater processes found running" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Read-Host "Press Enter to continue"
}

function Show-UpdateLog {
    Write-Host "📋 Recent Update Log:" -ForegroundColor Green
    Write-Host "=" * 50 -ForegroundColor Cyan
    
    $logFile = Join-Path $ProjectPath "update_log.txt"
    if (Test-Path $logFile) {
        Get-Content $logFile | Select-Object -Last 20
    } else {
        Write-Host "ℹ️  No log file found. Auto-updater hasn't run yet." -ForegroundColor Yellow
    }
    
    Write-Host "=" * 50 -ForegroundColor Cyan
    Write-Host ""
    Read-Host "Press Enter to continue"
}

function Show-SystemStatus {
    Write-Host "📊 System Status Check:" -ForegroundColor Green
    Write-Host "=" * 50 -ForegroundColor Cyan
    
    # Check if auto-updater is running
    $runningProcesses = Get-Process | Where-Object { $_.ProcessName -eq "python" }
    if ($runningProcesses) {
        Write-Host "✅ Python processes found: $($runningProcesses.Count)" -ForegroundColor Green
    } else {
        Write-Host "❌ No Python processes running" -ForegroundColor Red
    }
    
    # Check data file
    $dataFile = Join-Path $ProjectPath "data\latest_market_data.csv"
    if (Test-Path $dataFile) {
        $lastWrite = (Get-Item $dataFile).LastWriteTime
        Write-Host "✅ Market data file exists" -ForegroundColor Green
        Write-Host "📅 Last updated: $lastWrite" -ForegroundColor Yellow
    } else {
        Write-Host "❌ Market data file not found" -ForegroundColor Red
    }
    
    # Check log file
    $logFile = Join-Path $ProjectPath "update_log.txt"
    if (Test-Path $logFile) {
        Write-Host "✅ Update log exists" -ForegroundColor Green
        $lastEntry = Get-Content $logFile | Select-Object -Last 1
        Write-Host "📝 Last entry: $lastEntry" -ForegroundColor Yellow
    } else {
        Write-Host "❌ No update log found" -ForegroundColor Red
    }
    
    # Check Git status
    try {
        $gitStatus = & git status --porcelain 2>$null
        if ($LASTEXITCODE -eq 0) {
            if ($gitStatus) {
                Write-Host "⚠️  Git: Uncommitted changes found" -ForegroundColor Yellow
            } else {
                Write-Host "✅ Git: Working directory clean" -ForegroundColor Green
            }
        } else {
            Write-Host "❌ Git: Not a git repository or git not installed" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "❌ Git: Error checking status" -ForegroundColor Red
    }
    
    Write-Host "=" * 50 -ForegroundColor Cyan
    Write-Host ""
    Read-Host "Press Enter to continue"
}

function Install-Dependencies {
    Write-Host "⚙️  Installing Python dependencies..." -ForegroundColor Green
    Write-Host ""
    
    try {
        & pip install schedule yfinance pandas numpy logging pathlib
        Write-Host "✅ Dependencies installed successfully!" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Error installing dependencies: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "🔧 Make sure pip is installed and in PATH" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Read-Host "Press Enter to continue"
}

function Set-GitConfiguration {
    Write-Host "🔧 Setting up Git configuration..." -ForegroundColor Green
    Write-Host ""
    
    # Check if already configured
    try {
        $userName = & git config user.name 2>$null
        $userEmail = & git config user.email 2>$null
        
        if ($userName -and $userEmail) {
            Write-Host "✅ Git already configured:" -ForegroundColor Green
            Write-Host "   Name: $userName" -ForegroundColor Yellow
            Write-Host "   Email: $userEmail" -ForegroundColor Yellow
        } else {
            Write-Host "⚠️  Git not configured. Setting up..." -ForegroundColor Yellow
            
            $name = Read-Host "Enter your name"
            $email = Read-Host "Enter your email"
            
            & git config user.name $name
            & git config user.email $email
            
            Write-Host "✅ Git configured successfully!" -ForegroundColor Green
        }
        
        # Initialize repository if not already
        if (-not (Test-Path ".git")) {
            Write-Host "🔧 Initializing Git repository..." -ForegroundColor Yellow
            & git init
            & git add .
            & git commit -m "Initial commit: Stock Prediction Hub setup"
            Write-Host "✅ Git repository initialized!" -ForegroundColor Green
        }
    }
    catch {
        Write-Host "❌ Error configuring Git: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Write-Host ""
    Read-Host "Press Enter to continue"
}

# Main execution
switch ($Action.ToLower()) {
    "start" { Start-AutoUpdater }
    "update" { Invoke-SingleUpdate }
    "stop" { Stop-AutoUpdater }
    "log" { Show-UpdateLog }
    "status" { Show-SystemStatus }
    "install" { Install-Dependencies }
    "git" { Set-GitConfiguration }
    default {
        # Interactive menu
        do {
            Show-Menu
            $choice = Read-Host "Enter your choice (1-8)"
            
            switch ($choice) {
                "1" { Start-AutoUpdater }
                "2" { Invoke-SingleUpdate }
                "3" { Stop-AutoUpdater }
                "4" { Show-UpdateLog }
                "5" { Show-SystemStatus }
                "6" { Install-Dependencies }
                "7" { Set-GitConfiguration }
                "8" { 
                    Write-Host ""
                    Write-Host "👋 Thank you for using Stock Prediction Hub!" -ForegroundColor Green
                    exit 
                }
                default { 
                    Write-Host "❌ Invalid choice. Please try again." -ForegroundColor Red
                    Start-Sleep -Seconds 2
                }
            }
        } while ($choice -ne "8")
    }
}
