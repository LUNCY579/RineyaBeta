# Offline end-to-end smoke test.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra)
. "$PSScriptRoot\_common.ps1"
Invoke-Lab smoke --all @Extra
