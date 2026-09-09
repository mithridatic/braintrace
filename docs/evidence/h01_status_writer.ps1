function Write-H01Status {
    param([string]$Path, [object]$Record)
    $temporary = $Path + '.' + [Guid]::NewGuid().ToString('N') + '.tmp'
    try {
        $json = $Record | ConvertTo-Json -Depth 12
        [IO.File]::WriteAllText($temporary, $json)
        if ([IO.File]::Exists($Path)) {
            [IO.File]::Replace($temporary, $Path, [NullString]::Value)
        } else {
            [IO.File]::Move($temporary, $Path)
        }
        return $true
    } catch [IO.IOException] {
        # A reader may deny replacement. Status publication is not a stop gate.
        return $false
    } finally {
        if ([IO.File]::Exists($temporary)) {
            [IO.File]::Delete($temporary)
        }
    }
}
