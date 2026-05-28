$proc = Get-Process -Id 29736
$uptime = (Get-Date) - $proc.StartTime
Write-Host "Uptime: $($uptime.ToString('hh\:mm\:ss'))"
Write-Host "CPU Time: $($proc.TotalProcessorTime.TotalSeconds) seconds"
Write-Host "Threads: $($proc.Threads.Count)"
