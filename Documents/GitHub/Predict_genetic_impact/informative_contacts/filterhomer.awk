#!/usr/bin/awk -f

BEGIN {
}
{
    if ($1 ~ /^#/) { next;}
    print $2,$3,$4,$1,$8
    if ($5 >= 10) { print; next;}   ## MAPQ > 10
}
END {
}

