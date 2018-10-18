#pragma once
#include "di.hpp"

class DIConfig: public boost::di::config
{
public:
    //static auto policies(...) noexcept {
    //    namespace di = boost::di;
    //    using namespace di::policies;
    //    return di::make_policies(constructible(di::policies::is_bound<di::_>{}));
    //}
};
