#include "PageTable.hpp"
#include <cassert>

PageTable::PageTable(size_t num_pages){
    resize(num_pages);
}

void PageTable::resize(size_t num_pages){
    entries.resize(num_pages);
}

size_t PageTable::size() const {
    return entries.size();
}

PageMeta& PageTable::getEntry(u64 vpn){
    assert(vpn < entries.size());
    return entries[vpn];
}

const PageMeta& PageTable::getEntry(u64 vpn) const {
    assert(vpn < entries.size());
    return entries[vpn];
}

void PageTable::updateEntry(u64 vpn, const PageMeta& entry){
    assert(vpn < entries.size());
    entries[vpn] = entry;
}

void PageTable::clearReference(u64 vpn){
    assert(vpn < entries.size());
    entries[vpn].reference = false;
}