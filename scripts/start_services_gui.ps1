param(
    [switch]$SkipBackend,
    [switch]$SkipFrontend,
    [switch]$SkipBrowser
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Add-Type -AssemblyName PresentationFramework, PresentationCore, WindowsBase
Add-Type -AssemblyName System.Drawing

function New-AiptLauncherBackgroundImage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$OutputPath,
        [int]$Width = 920,
        [int]$Height = 520
    )

    $bmp = New-Object System.Drawing.Bitmap $Width, $Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias

    try {
        $rect = New-Object System.Drawing.Rectangle 0, 0, $Width, $Height
        $c1 = [System.Drawing.Color]::FromArgb(255, 10, 18, 32)
        $c2 = [System.Drawing.Color]::FromArgb(255, 18, 48, 70)
        $bgBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush $rect, $c1, $c2, 45.0
        $g.FillRectangle($bgBrush, $rect)
        $bgBrush.Dispose()

        $gridPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(24, 255, 255, 255)), 1
        for ($x = 0; $x -le $Width; $x += 40) { $g.DrawLine($gridPen, $x, 0, $x, $Height) }
        for ($y = 0; $y -le $Height; $y += 40) { $g.DrawLine($gridPen, 0, $y, $Width, $y) }
        $gridPen.Dispose()

        $titleFont = New-Object System.Drawing.Font "Segoe UI", 54, ([System.Drawing.FontStyle]::Bold)
        $titleBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(28, 255, 255, 255))
        $g.DrawString("AIPT", $titleFont, $titleBrush, (New-Object System.Drawing.PointF 520, 26))
        $titleBrush.Dispose()
        $titleFont.Dispose()

        function Draw-DefectCard {
            param(
                [int]$X,
                [int]$Y,
                [string]$Name,
                [System.Drawing.Color]$Accent,
                [string]$Pattern
            )
            $cardW = 250
            $cardH = 140
            $pad = 12

            $cardRect = New-Object System.Drawing.Rectangle $X, $Y, $cardW, $cardH
            $cardFill = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(160, 9, 14, 24))
            $g.FillRectangle($cardFill, $cardRect)
            $cardFill.Dispose()

            $borderPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(90, 255, 255, 255)), 1
            $g.DrawRectangle($borderPen, $cardRect)
            $borderPen.Dispose()

            $imgRect = New-Object System.Drawing.Rectangle ($X + $pad), ($Y + $pad + 18), ($cardW - 2 * $pad), ($cardH - 2 * $pad - 18)
            $imgBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush $imgRect, ([System.Drawing.Color]::FromArgb(255, 44, 52, 60)), ([System.Drawing.Color]::FromArgb(255, 22, 28, 34)), 90.0
            $g.FillRectangle($imgBrush, $imgRect)
            $imgBrush.Dispose()

            $noisePen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(30, 255, 255, 255)), 1
            for ($i = 0; $i -lt 24; $i++) {
                $yy = $imgRect.Top + 2 + ($i * 4)
                $g.DrawLine($noisePen, $imgRect.Left + 2, $yy, $imgRect.Right - 2, $yy)
            }
            $noisePen.Dispose()

            switch ($Pattern) {
                "scratch" {
                    $p = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(220, 255, 255, 255)), 3
                    $g.DrawLine($p, $imgRect.Left + 14, $imgRect.Bottom - 22, $imgRect.Right - 28, $imgRect.Top + 26)
                    $p.Dispose()
                }
                "dent" {
                    $centerX = [int](($imgRect.Left + $imgRect.Right) / 2)
                    $centerY = [int](($imgRect.Top + $imgRect.Bottom) / 2)
                    $r = 22
                    $b1 = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(140, 0, 0, 0))
                    $g.FillEllipse($b1, $centerX - $r, $centerY - $r, 2 * $r, 2 * $r)
                    $b1.Dispose()
                    $b2 = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(80, 255, 255, 255))
                    $g.FillEllipse($b2, $centerX - $r + 6, $centerY - $r + 6, 2 * ($r - 6), 2 * ($r - 6))
                    $b2.Dispose()
                }
                "crack" {
                    $p = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(220, 240, 240, 240)), 2
                    $pts = @(
                        (New-Object System.Drawing.Point ($imgRect.Left + 18), ($imgRect.Top + 20)),
                        (New-Object System.Drawing.Point ($imgRect.Left + 54), ($imgRect.Top + 44)),
                        (New-Object System.Drawing.Point ($imgRect.Left + 78), ($imgRect.Top + 34)),
                        (New-Object System.Drawing.Point ($imgRect.Left + 112), ($imgRect.Top + 62)),
                        (New-Object System.Drawing.Point ($imgRect.Left + 146), ($imgRect.Top + 52)),
                        (New-Object System.Drawing.Point ($imgRect.Left + 176), ($imgRect.Top + 86)),
                        (New-Object System.Drawing.Point ($imgRect.Left + 212), ($imgRect.Top + 74))
                    )
                    $g.DrawLines($p, $pts)
                    $p.Dispose()
                }
            }

            $boxPen = New-Object System.Drawing.Pen $Accent, 2
            $boxPen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
            $bbox = New-Object System.Drawing.Rectangle ($imgRect.Left + 14), ($imgRect.Top + 18), ($imgRect.Width - 64), ($imgRect.Height - 44)
            $g.DrawRectangle($boxPen, $bbox)
            $boxPen.Dispose()

            $labelFont = New-Object System.Drawing.Font "Segoe UI Semibold", 12, ([System.Drawing.FontStyle]::Bold)
            $labelBrush = New-Object System.Drawing.SolidBrush $Accent
            $g.DrawString($Name, $labelFont, $labelBrush, (New-Object System.Drawing.PointF ($X + $pad), ($Y + 10)))
            $labelBrush.Dispose()
            $labelFont.Dispose()
        }

        Draw-DefectCard -X 628 -Y 116 -Name "Scratch" -Accent ([System.Drawing.Color]::FromArgb(255, 34, 197, 94)) -Pattern "scratch"
        Draw-DefectCard -X 628 -Y 280 -Name "Dent" -Accent ([System.Drawing.Color]::FromArgb(255, 59, 130, 246)) -Pattern "dent"
        Draw-DefectCard -X 628 -Y 368 -Name "Crack" -Accent ([System.Drawing.Color]::FromArgb(255, 239, 68, 68)) -Pattern "crack"

        $hintFont = New-Object System.Drawing.Font "Segoe UI", 12, ([System.Drawing.FontStyle]::Regular)
        $hintBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(120, 255, 255, 255))
        $g.DrawString("Industrial Defect Detection • Annotation • Training • Inference", $hintFont, $hintBrush, (New-Object System.Drawing.PointF 36, 478))
        $hintBrush.Dispose()
        $hintFont.Dispose()

        $bmp.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $g.Dispose()
        $bmp.Dispose()
    }
}

$root = Split-Path -Parent $PSScriptRoot
$logsDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$xaml = @"
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="AIPT 启动器"
        Width="920" Height="520"
        WindowStartupLocation="CenterScreen"
        ResizeMode="NoResize"
        Background="#0B1220">
  <Grid>
    <Image x:Name="BgImage" Stretch="Fill" />
    <Border Margin="22"
            Width="462"
            HorizontalAlignment="Left"
            Background="#D80B1220"
            CornerRadius="14"
            BorderBrush="#33FFFFFF"
            BorderThickness="1"
            Padding="16">
      <Grid>
        <Grid.RowDefinitions>
          <RowDefinition Height="Auto"/>
          <RowDefinition Height="Auto"/>
          <RowDefinition Height="Auto"/>
          <RowDefinition Height="*"/>
          <RowDefinition Height="Auto"/>
          <RowDefinition Height="Auto"/>
        </Grid.RowDefinitions>

        <StackPanel Grid.Row="0">
          <TextBlock Text="AIPT 平台启动中" FontSize="20" FontWeight="SemiBold" Foreground="#E8FFFFFF"/>
          <TextBlock x:Name="StatusText" Text="准备启动..." Margin="0,6,0,0" FontSize="12" Foreground="#B8FFFFFF"/>
        </StackPanel>

        <StackPanel Grid.Row="1" Orientation="Horizontal" Margin="0,12,0,0">
          <CheckBox x:Name="ChkBackend" Content="启动后端" IsChecked="True" Foreground="#E8FFFFFF" Margin="0,0,14,0"/>
          <CheckBox x:Name="ChkFrontend" Content="启动前端" IsChecked="True" Foreground="#E8FFFFFF" Margin="0,0,14,0"/>
          <CheckBox x:Name="ChkBrowser" Content="打开浏览器" IsChecked="True" Foreground="#E8FFFFFF"/>
        </StackPanel>

        <ProgressBar x:Name="Progress" Grid.Row="2" Height="10" Margin="0,12,0,0" Minimum="0" Maximum="8"
                     Background="#1AFFFFFF" Foreground="#22C55E" BorderThickness="0"/>

        <Grid Grid.Row="3" Margin="0,12,0,0">
          <Grid.RowDefinitions>
            <RowDefinition Height="160"/>
            <RowDefinition Height="*"/>
          </Grid.RowDefinitions>
          <Border Grid.Row="0" Background="#0FFFFFFF" CornerRadius="10" Padding="10">
            <ListBox x:Name="StepsList" Background="Transparent" BorderThickness="0" Foreground="#E8FFFFFF" FontSize="12"/>
          </Border>
          <Border Grid.Row="1" Background="#0FFFFFFF" CornerRadius="10" Padding="10" Margin="0,12,0,0">
            <TextBox x:Name="LogBox" Background="Transparent" BorderThickness="0" Foreground="#C8FFFFFF"
                     FontFamily="Consolas" FontSize="11" IsReadOnly="True" TextWrapping="Wrap"
                     VerticalScrollBarVisibility="Auto"/>
          </Border>
        </Grid>

        <StackPanel Grid.Row="4" Orientation="Horizontal" HorizontalAlignment="Right" Margin="0,12,0,0">
          <Button x:Name="BtnOpen" Content="打开平台" Padding="12,6" Margin="0,0,10,0" IsEnabled="False"/>
          <Button x:Name="BtnStop" Content="停止服务" Padding="12,6" Margin="0,0,10,0" IsEnabled="False"/>
          <Button x:Name="BtnClose" Content="关闭" Padding="12,6" IsEnabled="False"/>
        </StackPanel>

        <TextBlock Grid.Row="5" Margin="0,10,0,0" Foreground="#88FFFFFF" FontSize="11"
                   Text="提示：日志位于 ./logs，默认端口：后端 8000 / 前端 5173"/>
      </Grid>
    </Border>
  </Grid>
</Window>
"@

$window = [Windows.Markup.XamlReader]::Load((New-Object System.Xml.XmlNodeReader ([xml]$xaml)))

$bgImage = $window.FindName("BgImage")
$statusText = $window.FindName("StatusText")
$progress = $window.FindName("Progress")
$stepsList = $window.FindName("StepsList")
$logBox = $window.FindName("LogBox")
$btnOpen = $window.FindName("BtnOpen")
$btnStop = $window.FindName("BtnStop")
$btnClose = $window.FindName("BtnClose")
$chkBackend = $window.FindName("ChkBackend")
$chkFrontend = $window.FindName("ChkFrontend")
$chkBrowser = $window.FindName("ChkBrowser")

$tmpBg = Join-Path ([System.IO.Path]::GetTempPath()) "aipt_launcher_bg.png"
try {
    New-AiptLauncherBackgroundImage -OutputPath $tmpBg | Out-Null
    if (Test-Path $tmpBg) {
        $bmp = New-Object System.Windows.Media.Imaging.BitmapImage
        $bmp.BeginInit()
        $bmp.CacheOption = [System.Windows.Media.Imaging.BitmapCacheOption]::OnLoad
        $bmp.UriSource = [Uri]$tmpBg
        $bmp.EndInit()
        $bgImage.Source = $bmp
    }
} catch {
    # Best effort: keep default background.
}

$stepNames = @(
    "检查运行环境（Python / Node）",
    "校验/安装后端依赖",
    "释放端口（8000 / 5173）",
    "启动后端服务",
    "等待后端就绪",
    "校验/安装前端依赖",
    "启动前端服务",
    "等待前端就绪并打开平台"
)

$steps = New-Object System.Collections.ObjectModel.ObservableCollection[string]
for ($i = 0; $i -lt $stepNames.Count; $i++) {
    $steps.Add(("[ ] {0}" -f $stepNames[$i]))
}
$stepsList.ItemsSource = $steps
$progress.Maximum = $stepNames.Count

function Add-UILog {
    param([string]$Message)
    $ts = Get-Date -Format "HH:mm:ss"
    $logBox.AppendText("[$ts] $Message`r`n")
    $logBox.ScrollToEnd()
}

function Stop-AiptServices {
    param([int[]]$Ports = @(8000, 5173))
    foreach ($p in $Ports) {
        try {
            $conn = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($null -ne $conn) {
                Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
            }
        } catch {
            # ignore
        }
    }
}

$btnClose.Add_Click({ $window.Close() })
$btnOpen.Add_Click({
    try { Start-Process "http://localhost:5173" } catch {}
})
$btnStop.Add_Click({
    Add-UILog "Stopping services on ports 8000/5173..."
    Stop-AiptServices
    Add-UILog "Services stopped."
})

$sync = [hashtable]::Synchronized(@{
    Root = $root
    LogsDir = $logsDir
    SkipBackend = [bool]$SkipBackend
    SkipFrontend = [bool]$SkipFrontend
    SkipBrowser = [bool]$SkipBrowser
    Window = $window
    Steps = $steps
    Progress = $progress
    StatusText = $statusText
    LogBox = $logBox
    BtnOpen = $btnOpen
    BtnStop = $btnStop
    BtnClose = $btnClose
    ChkBackend = $chkBackend
    ChkFrontend = $chkFrontend
    ChkBrowser = $chkBrowser
})

$backgroundScript = {
    param($sync)

    $ErrorActionPreference = "Stop"
    Set-StrictMode -Version Latest

    function UIAction {
        param([scriptblock]$Action)
        $sync.Window.Dispatcher.Invoke([Action]$Action) | Out-Null
    }

    function UIValue {
        param([scriptblock]$Action)
        return $sync.Window.Dispatcher.Invoke([Func[object]]{ & $Action })
    }

    function Set-Step {
        param(
            [int]$Index,
            [ValidateSet("pending", "running", "ok", "skip", "err")]
            [string]$State,
            [string]$Suffix = ""
        )
        UIAction {
            $prefix = switch ($State) {
                "pending" { "[ ]" }
                "running" { "[...]" }
                "ok" { "[OK]" }
                "skip" { "[SKIP]" }
                "err" { "[ERR]" }
            }
            $name = $sync.Steps[$Index] -replace "^\\[[^\\]]+\\]", $prefix
            if ($Suffix) {
                $name = ($name -replace "\\s+-\\s+.*$", "") + " - " + $Suffix
            }
            $sync.Steps[$Index] = $name
        }
    }

    function Set-Status([string]$Text) {
        UIAction { $sync.StatusText.Text = $Text }
    }

    function Log([string]$Message) {
        UIAction {
            $ts = Get-Date -Format "HH:mm:ss"
            $sync.LogBox.AppendText("[$ts] $Message`r`n")
            $sync.LogBox.ScrollToEnd()
        }
    }

    function Tick-Progress {
        UIAction { $sync.Progress.Value = [Math]::Min($sync.Progress.Maximum, $sync.Progress.Value + 1) }
    }

    function Invoke-Process {
        param(
            [Parameter(Mandatory = $true)]
            [string]$FilePath,
            [string[]]$ArgumentList = @(),
            [string]$WorkingDirectory = $sync.Root,
            [string]$StdOutPath = $null,
            [string]$StdErrPath = $null
        )

        $command = Build-CmdCommandLine -FilePath $FilePath -ArgumentList $ArgumentList
        $command = Add-CmdRedirections -CommandLine $command -StdOutPath $StdOutPath -StdErrPath $StdErrPath
        $p = Start-CmdProcess -CommandLine $command -WorkingDirectory $WorkingDirectory -Wait
        if ($p.ExitCode -ne 0) {
            throw "Command failed ($FilePath) exit=$($p.ExitCode)"
        }
        return $p
    }

    function Get-NvidiaCudaVersion {
        try {
            $exe = (Get-Command nvidia-smi -ErrorAction Stop).Path
        } catch {
            return $null
        }

        try {
            $out = & $exe 2>$null
            if ($LASTEXITCODE -ne 0) { return $null }
            $text = ($out -join "`n")
            if ($text -match "CUDA Version:\\s*([0-9]+\\.[0-9]+)") {
                return $Matches[1]
            }
        } catch {
            return $null
        }
        return $null
    }

    function Ensure-CudaTorch {
        param(
            [Parameter(Mandatory = $true)]
            [string]$PythonExe,
            [Parameter(Mandatory = $true)]
            [string]$StdOutPath,
            [Parameter(Mandatory = $true)]
            [string]$StdErrPath
        )

        $cudaText = Get-NvidiaCudaVersion
        if (-not $cudaText) { return }

        $cudaVer = $null
        try { $cudaVer = [version]$cudaText } catch { $cudaVer = $null }
        if (-not $cudaVer) { return }

        $torchVer = $null
        $cudaAvailable = $false
        $torchCuda = $null
        try {
            $probe = & $PythonExe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(getattr(torch.version,'cuda',None))" 2>$null
            $probeLines = @($probe)
            if ($LASTEXITCODE -eq 0 -and $probeLines.Count -ge 2) {
                $torchVer = ([string]$probeLines[0]).Trim()
                $cudaAvailable = ([string]$probeLines[1]).Trim().ToLowerInvariant() -eq "true"
                if ($probeLines.Count -ge 3) {
                    $torchCuda = ([string]$probeLines[2]).Trim()
                }
            }
        } catch {
            $cudaAvailable = $false
        }

        if ($cudaAvailable) {
            Log "PyTorch CUDA is available (torch=$torchVer, cuda=$torchCuda)."
            return
        }

        $indexUrl = $null
        if ($cudaVer.Major -gt 12 -or ($cudaVer.Major -eq 12 -and $cudaVer.Minor -ge 1)) {
            $indexUrl = "https://download.pytorch.org/whl/cu121"
        } elseif ($cudaVer.Major -eq 12 -and $cudaVer.Minor -eq 0) {
            $indexUrl = "https://download.pytorch.org/whl/cu118"
        } elseif ($cudaVer.Major -eq 11 -and $cudaVer.Minor -ge 8) {
            $indexUrl = "https://download.pytorch.org/whl/cu118"
        }

        if (-not $indexUrl) {
            Log "Detected NVIDIA GPU (CUDA $cudaText) but no compatible PyTorch CUDA wheels configured; skip auto-install."
            return
        }

        Log "Detected NVIDIA GPU (CUDA $cudaText) but PyTorch CUDA is not available (torch=$torchVer). Installing CUDA-enabled PyTorch..."
        try {
            Invoke-Process -FilePath $PythonExe -ArgumentList @("-m", "pip", "install", "--upgrade", "--index-url", $indexUrl, "torch==2.2.*", "torchvision==0.17.*") -StdOutPath $StdOutPath -StdErrPath $StdErrPath | Out-Null
        } catch {
            Log "WARNING: Failed to install CUDA-enabled PyTorch. Training may run CPU-only. See logs: $StdOutPath / $StdErrPath"
            return
        }

        try {
            $probe2 = & $PythonExe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(getattr(torch.version,'cuda',None))" 2>$null
            $probe2Lines = @($probe2)
            if ($LASTEXITCODE -eq 0 -and $probe2Lines.Count -ge 2) {
                $torchVer2 = ([string]$probe2Lines[0]).Trim()
                $cudaOk2 = ([string]$probe2Lines[1]).Trim().ToLowerInvariant() -eq "true"
                $torchCuda2 = $null
                if ($probe2Lines.Count -ge 3) { $torchCuda2 = ([string]$probe2Lines[2]).Trim() }
                if ($cudaOk2) {
                    Log "CUDA-enabled PyTorch ready (torch=$torchVer2, cuda=$torchCuda2)."
                } else {
                    Log "CUDA-enabled PyTorch installed but CUDA still unavailable (torch=$torchVer2, cuda=$torchCuda2)."
                }
            }
        } catch {
            # ignore
        }
    }

    function Start-ProcessDetached {
        param(
            [Parameter(Mandatory = $true)]
            [string]$FilePath,
            [string[]]$ArgumentList = @(),
            [string]$WorkingDirectory = $sync.Root,
            [string]$StdOutPath = $null,
            [string]$StdErrPath = $null
        )
        $command = Build-CmdCommandLine -FilePath $FilePath -ArgumentList $ArgumentList
        $command = Add-CmdRedirections -CommandLine $command -StdOutPath $StdOutPath -StdErrPath $StdErrPath
        return Start-CmdProcess -CommandLine $command -WorkingDirectory $WorkingDirectory
    }

    function Quote-CmdValue {
        param([string]$Value)
        if ($null -eq $Value) { return '""' }
        if ($Value.Length -eq 0) { return '""' }

        if ($Value -match '[\s&()^|<>\"]') {
            $escaped = $Value -replace '"', '""'
            return '"' + $escaped + '"'
        }
        return $Value
    }

    function Build-CmdCommandLine {
        param(
            [Parameter(Mandatory = $true)]
            [string]$FilePath,
            [string[]]$ArgumentList = @()
        )
        $cmd = Quote-CmdValue -Value $FilePath
        foreach ($a in $ArgumentList) {
            $cmd += " " + (Quote-CmdValue -Value $a)
        }
        return $cmd.Trim()
    }

    function Add-CmdRedirections {
        param(
            [Parameter(Mandatory = $true)]
            [string]$CommandLine,
            [string]$StdOutPath = $null,
            [string]$StdErrPath = $null
        )

        $outPath = $StdOutPath
        $errPath = $StdErrPath
        if ($outPath -and $errPath) {
            try {
                $outFull = [System.IO.Path]::GetFullPath($outPath)
                $errFull = [System.IO.Path]::GetFullPath($errPath)
                if ($outFull.Equals($errFull, [System.StringComparison]::OrdinalIgnoreCase)) {
                    $errPath = "$outPath.err"
                }
            } catch {
                # ignore
            }
        }

        if (-not $outPath) { $outPath = "NUL" }
        if (-not $errPath) { $errPath = "NUL" }

        if ($outPath -ne "NUL") {
            New-Item -ItemType Directory -Force -Path (Split-Path $outPath -Parent) | Out-Null
        }
        if ($errPath -ne "NUL") {
            New-Item -ItemType Directory -Force -Path (Split-Path $errPath -Parent) | Out-Null
        }

        $redir = " 1>$(Quote-CmdValue -Value $outPath) 2>$(Quote-CmdValue -Value $errPath)"
        return ($CommandLine + $redir)
    }

    function Start-CmdProcess {
        param(
            [Parameter(Mandatory = $true)]
            [string]$CommandLine,
            [string]$WorkingDirectory = $sync.Root,
            [switch]$Wait
        )
        $cmdExe = $env:ComSpec
        if (-not $cmdExe) { $cmdExe = "cmd.exe" }

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $cmdExe
        $psi.WorkingDirectory = $WorkingDirectory
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.Arguments = "/d /c $CommandLine"

        $proc = New-Object System.Diagnostics.Process
        $proc.StartInfo = $psi
        if (-not $proc.Start()) {
            throw "Failed to start command: $CommandLine"
        }
        if ($Wait) {
            $proc.WaitForExit()
        }
        return $proc
    }

    function Ensure-PortsFree([int[]]$Ports) {
        foreach ($p in $Ports) {
            try {
                $conn = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
                if ($null -ne $conn) {
                    Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
                    Log "Freed port $p (killed PID $($conn.OwningProcess))."
                }
            } catch {
                Log "Port free check failed for ${p}: $($_.Exception.Message)"
            }
        }
    }

    function Wait-HttpOk {
        param(
            [Parameter(Mandatory = $true)]
            [string]$Url,
            [int]$TimeoutSeconds = 60,
            [switch]$UseBasicParsing
        )
        $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
        $lastErr = $null
        while ((Get-Date) -lt $deadline) {
            try {
                if ($UseBasicParsing) {
                    Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2 -ErrorAction Stop | Out-Null
                } else {
                    Invoke-RestMethod -Uri $Url -TimeoutSec 2 -ErrorAction Stop | Out-Null
                }
                return
            } catch {
                $lastErr = $_
                Start-Sleep -Seconds 1
            }
        }
        throw "Timeout waiting for $Url ($($lastErr.Exception.Message))"
    }

    Set-Status "准备启动..."
    Start-Sleep -Milliseconds 800

    $cfg = UIValue {
        [pscustomobject]@{
            StartBackend = -not $sync.SkipBackend -and [bool]$sync.ChkBackend.IsChecked
            StartFrontend = -not $sync.SkipFrontend -and [bool]$sync.ChkFrontend.IsChecked
            OpenBrowser = -not $sync.SkipBrowser -and [bool]$sync.ChkBrowser.IsChecked
        }
    }

    UIAction {
        $sync.ChkBackend.IsEnabled = $false
        $sync.ChkFrontend.IsEnabled = $false
        $sync.ChkBrowser.IsEnabled = $false
    }

    $backendPython = "python"
    $venvPython = Join-Path $sync.Root ".venv\\Scripts\\python.exe"
    $preferred = "D:\\APPS\\Environments\\Anaconda3\\envs\\vape_gpu\\python.exe"
    if (Test-Path $venvPython) {
        $backendPython = $venvPython
    } elseif (Test-Path $preferred) {
        $backendPython = $preferred
    }

    $nodeExe = "node.exe"
    $npmCmd = "npm.cmd"
    try {
        $nodeExe = (Get-Command node.exe -ErrorAction Stop).Path
    } catch {
        # fallback to PATH resolution
        $nodeExe = "node.exe"
    }
    try {
        $npmCmd = (Get-Command npm.cmd -ErrorAction Stop).Path
    } catch {
        # fallback: best-effort; avoid npm.ps1 when possible
        try {
            $npmCmd = (Get-Command npm -ErrorAction Stop).Path
        } catch {
            $npmCmd = "npm.cmd"
        }
    }
    try {
        if ($npmCmd -and $npmCmd.ToLowerInvariant().EndsWith(".ps1")) {
            $candidate = Join-Path (Split-Path $npmCmd -Parent) "npm.cmd"
            if (Test-Path $candidate) {
                $npmCmd = $candidate
            }
        }
    } catch {
        # ignore
    }

    $backendLog = Join-Path $sync.LogsDir "backend.log"
    $backendErr = Join-Path $sync.LogsDir "backend.err.log"
    $backendInstallLog = Join-Path $sync.LogsDir "backend_install.log"
    $backendInstallErr = Join-Path $sync.LogsDir "backend_install.err.log"
    $envPythonLog = Join-Path $sync.LogsDir "env_python.log"
    $envPythonErr = Join-Path $sync.LogsDir "env_python.err.log"
    $envNodeLog = Join-Path $sync.LogsDir "env_node.log"
    $envNodeErr = Join-Path $sync.LogsDir "env_node.err.log"
    $envNpmLog = Join-Path $sync.LogsDir "env_npm.log"
    $envNpmErr = Join-Path $sync.LogsDir "env_npm.err.log"
    $frontendLog = Join-Path $sync.LogsDir "frontend.log"
    $frontendErr = Join-Path $sync.LogsDir "frontend.err.log"
    $frontendInstallLog = Join-Path $sync.LogsDir "frontend_install.log"
    $frontendInstallErr = Join-Path $sync.LogsDir "frontend_install.err.log"

    try {
        Set-Status "检查运行环境..."
        Set-Step 0 running
        Log "Using backend python: $backendPython"
        Log "Using node: $nodeExe"
        Log "Using npm: $npmCmd"

        Invoke-Process -FilePath $backendPython -ArgumentList @("--version") -StdOutPath $envPythonLog -StdErrPath $envPythonErr | Out-Null
        Invoke-Process -FilePath $nodeExe -ArgumentList @("--version") -StdOutPath $envNodeLog -StdErrPath $envNodeErr | Out-Null
        Invoke-Process -FilePath $npmCmd -ArgumentList @("--version") -StdOutPath $envNpmLog -StdErrPath $envNpmErr | Out-Null

        Set-Step 0 ok
        Tick-Progress

        if (-not $cfg.StartBackend) {
            Set-Step 1 skip "后端已跳过"
            Set-Step 2 skip "后端已跳过"
            Set-Step 3 skip "后端已跳过"
            Set-Step 4 skip "后端已跳过"
            Tick-Progress; Tick-Progress; Tick-Progress; Tick-Progress
        } else {
            Set-Status "校验/安装后端依赖..."
            Set-Step 1 running

            $depOk = $true
            try {
                Invoke-Process -FilePath $backendPython -ArgumentList @("-c", "import fastapi, uvicorn, sqlalchemy; import PIL, numpy; import torch, ultralytics") -StdOutPath $backendInstallLog -StdErrPath $backendInstallErr | Out-Null
            } catch {
                $depOk = $false
            }

            if (-not $depOk) {
                Log "Missing backend dependencies; installing (stdout=$backendInstallLog, stderr=$backendInstallErr)..."
                Invoke-Process -FilePath $backendPython -ArgumentList @("-m", "pip", "--version") -StdOutPath $backendInstallLog -StdErrPath $backendInstallErr | Out-Null
                Invoke-Process -FilePath $backendPython -ArgumentList @("-m", "pip", "install", "-r", (Join-Path $sync.Root "backend\\requirements.txt")) -StdOutPath $backendInstallLog -StdErrPath $backendInstallErr | Out-Null
            }

            # If the machine has an NVIDIA GPU, prefer installing a CUDA-enabled PyTorch wheel.
            # This keeps the training module GPU-ready even though PyPI defaults to CPU wheels.
            try {
                Ensure-CudaTorch -PythonExe $backendPython -StdOutPath $backendInstallLog -StdErrPath $backendInstallErr
            } catch {
                Log "WARNING: CUDA PyTorch setup check failed: $($_.Exception.Message)"
            }
            Set-Step 1 ok
            Tick-Progress

            Set-Status "释放端口..."
            Set-Step 2 running
            Ensure-PortsFree -Ports @(8000, 5173)
            Set-Step 2 ok
            Tick-Progress

            Set-Status "启动后端服务..."
            Set-Step 3 running
            $uvicornArgs = @(
                "-m", "uvicorn",
                "backend.main:app",
                "--reload",
                "--reload-dir", "backend",
                "--host", "127.0.0.1",
                "--port", "8000"
            )
            Start-ProcessDetached -FilePath $backendPython -ArgumentList $uvicornArgs -StdOutPath $backendLog -StdErrPath $backendErr | Out-Null
            Set-Step 3 ok
            Tick-Progress

            Set-Status "等待后端就绪..."
            Set-Step 4 running
            Wait-HttpOk -Url "http://127.0.0.1:8000/health" -TimeoutSeconds 60
            Wait-HttpOk -Url "http://127.0.0.1:8000/datasets" -TimeoutSeconds 60
            Set-Step 4 ok
            Tick-Progress
        }

        if (-not $cfg.StartFrontend) {
            Set-Step 5 skip "前端已跳过"
            Set-Step 6 skip "前端已跳过"
            Set-Step 7 skip "前端已跳过"
            Tick-Progress; Tick-Progress; Tick-Progress
        } else {
            Set-Status "校验/安装前端依赖..."
            Set-Step 5 running

            $nodeModules = Join-Path $sync.Root "frontend\\node_modules"
            if (-not (Test-Path $nodeModules)) {
                Log "node_modules not found; running npm install (stdout=$frontendInstallLog, stderr=$frontendInstallErr)..."
                Invoke-Process -FilePath $npmCmd -ArgumentList @("install") -WorkingDirectory (Join-Path $sync.Root "frontend") -StdOutPath $frontendInstallLog -StdErrPath $frontendInstallErr | Out-Null
            }
            Set-Step 5 ok
            Tick-Progress

            Set-Status "启动前端服务..."
            Set-Step 6 running
            Start-ProcessDetached -FilePath $npmCmd -ArgumentList @("run", "dev") -WorkingDirectory (Join-Path $sync.Root "frontend") -StdOutPath $frontendLog -StdErrPath $frontendErr | Out-Null
            Set-Step 6 ok
            Tick-Progress

            Set-Status "等待前端就绪..."
            Set-Step 7 running
            Wait-HttpOk -Url "http://localhost:5173" -TimeoutSeconds 60 -UseBasicParsing

            if ($cfg.OpenBrowser) {
                try {
                    Start-Process "http://localhost:5173" | Out-Null
                    Set-Step 7 ok "已打开浏览器"
                } catch {
                    Set-Step 7 ok "已启动（浏览器打开失败）"
                }
            } else {
                Set-Step 7 ok "已启动"
            }
            Tick-Progress
        }

        Set-Status "启动完成（即将自动关闭）"
        UIAction {
            $sync.BtnOpen.IsEnabled = $true
            $sync.BtnStop.IsEnabled = $true
            $sync.BtnClose.IsEnabled = $true

            try {
                $sync.CloseTimer = New-Object System.Windows.Threading.DispatcherTimer
                $sync.CloseTimer.Interval = [TimeSpan]::FromSeconds(2)
                $sync.CloseTimer.Add_Tick({
                    try { $sync.CloseTimer.Stop() } catch {}
                    try { $sync.Window.Close() } catch {}
                })
                $sync.CloseTimer.Start()
            } catch {
                # ignore
            }
        }
        Log "DONE. Backend: http://localhost:8000/docs  Frontend: http://localhost:5173"
    } catch {
        $err = $_
        Log "FAILED: $($err.Exception.Message)"
        Set-Status "启动失败（请查看日志）"
        UIAction {
            $sync.BtnStop.IsEnabled = $true
            $sync.BtnClose.IsEnabled = $true
        }
        throw
    }
}

$rs = [runspacefactory]::CreateRunspace()
$rs.ApartmentState = "MTA"
$rs.ThreadOptions = "ReuseThread"
$rs.Open()
$ps = [powershell]::Create()
$ps.Runspace = $rs
$null = $ps.AddScript($backgroundScript.ToString()).AddArgument($sync)
$async = $ps.BeginInvoke()

$window.Add_Closed({
    try { $ps.Stop() } catch {}
    try { $ps.Dispose() } catch {}
    try { $rs.Close() } catch {}
    try { $rs.Dispose() } catch {}
})

$null = $window.ShowDialog()

