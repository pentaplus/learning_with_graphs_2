for ($i=1; $i -le 10; $i++) {
	python -u .\eval_embeddings.py | Tee-Object -FilePath .\out.log | Write-Host
	
	$source_path = '..\results\weisfeiler_lehman\'
	
	$target_path = $source_path + $i.ToString() + '. iteration'
	if(-not (Test-Path $target_path)) {
		New-Item -ItemType directory -Path $target_path
	}
	
	Start-Sleep -s 1
	robocopy /mov $source_path $target_path
	robocopy /mov . $target_path out.log
	Start-Sleep -s 1
}