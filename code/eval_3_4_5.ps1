python -u .\eval_embeddings_5.py | Tee-Object -FilePath .\out.log | Write-Host

$source_path = '..\results\weisfeiler_lehman\'

$target_path = $source_path + '1. iteration (5)'
if(-not (Test-Path $target_path)) {
	New-Item -ItemType directory -Path $target_path
}

Start-Sleep -s 1
	
robocopy /mov $source_path $target_path

robocopy /mov . $target_path out.log

Start-Sleep -s 1

python -u .\eval_embeddings_4.py | Tee-Object -FilePath .\out.log | Write-Host

$source_path = '..\results\weisfeiler_lehman\'

$target_path = $source_path + '2. iteration (4)'
if(-not (Test-Path $target_path)) {
	New-Item -ItemType directory -Path $target_path
}

Start-Sleep -s 1
	
robocopy /mov $source_path $target_path

robocopy /mov . $target_path out.log

Start-Sleep -s 1

python -u .\eval_embeddings_3.py | Tee-Object -FilePath .\out.log | Write-Host

$source_path = '..\results\weisfeiler_lehman\'

$target_path = $source_path + '3. iteration (3)'
if(-not (Test-Path $target_path)) {
	New-Item -ItemType directory -Path $target_path
}

Start-Sleep -s 1
	
robocopy /mov $source_path $target_path

robocopy /mov . $target_path out.log
