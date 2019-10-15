#!/usr/bin/awk -f

BEGIN {
}
{
    if ($1 ~ /^@/) { print; next;}
    if ($5 >= 10) { print; next;}   ## MAPQ > 10
}
END {
}

