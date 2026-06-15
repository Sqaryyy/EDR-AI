#!/bin/bash
# simulates ransomware-like behavior: create files, read them, rename them

mkdir -p /tmp/ransom_test

# create some fake "documents"
for i in {1..20}; do
    echo "sensitive data $i" > /tmp/ransom_test/doc_$i.txt
done

# read and "encrypt" (just rename with .enc extension)
for i in {1..20}; do
    cat /tmp/ransom_test/doc_$i.txt > /dev/null
    mv /tmp/ransom_test/doc_$i.txt /tmp/ransom_test/doc_$i.txt.enc
done

echo "done"