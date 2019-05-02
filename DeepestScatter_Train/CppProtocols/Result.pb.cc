// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: Result.proto

#include "Result.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// This is a temporary google only hack
#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
#include "third_party/protobuf/version.h"
#endif
// @@protoc_insertion_point(includes)

namespace Persistance {
class ResultDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<Result>
      _instance;
} _Result_default_instance_;
}  // namespace Persistance
namespace protobuf_Result_2eproto {
static void InitDefaultsResult() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::Persistance::_Result_default_instance_;
    new (ptr) ::Persistance::Result();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::Persistance::Result::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<0> scc_info_Result =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 0, InitDefaultsResult}, {}};

void InitDefaults() {
  ::google::protobuf::internal::InitSCC(&scc_info_Result.base);
}

::google::protobuf::Metadata file_level_metadata[1];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::Persistance::Result, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::Persistance::Result, light_intensity_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::Persistance::Result, is_converged_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::Persistance::Result)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::Persistance::_Result_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "Result.proto", schemas, file_default_instances, TableStruct::offsets,
      file_level_metadata, NULL, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_PROTOBUF_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 1);
}

void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
      "\n\014Result.proto\022\013Persistance\"7\n\006Result\022\027\n"
      "\017light_intensity\030\001 \001(\002\022\024\n\014is_converged\030\002"
      " \001(\010b\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 92);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "Result.proto", &protobuf_RegisterTypes);
}

void AddDescriptors() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, AddDescriptorsImpl);
}
// Force AddDescriptors() to be called at dynamic initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;
}  // namespace protobuf_Result_2eproto
namespace Persistance {

// ===================================================================

void Result::InitAsDefaultInstance() {
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int Result::kLightIntensityFieldNumber;
const int Result::kIsConvergedFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

Result::Result()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_Result_2eproto::scc_info_Result.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:Persistance.Result)
}
Result::Result(const Result& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::memcpy(&light_intensity_, &from.light_intensity_,
    static_cast<size_t>(reinterpret_cast<char*>(&is_converged_) -
    reinterpret_cast<char*>(&light_intensity_)) + sizeof(is_converged_));
  // @@protoc_insertion_point(copy_constructor:Persistance.Result)
}

void Result::SharedCtor() {
  ::memset(&light_intensity_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&is_converged_) -
      reinterpret_cast<char*>(&light_intensity_)) + sizeof(is_converged_));
}

Result::~Result() {
  // @@protoc_insertion_point(destructor:Persistance.Result)
  SharedDtor();
}

void Result::SharedDtor() {
}

void Result::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* Result::descriptor() {
  ::protobuf_Result_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_Result_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const Result& Result::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_Result_2eproto::scc_info_Result.base);
  return *internal_default_instance();
}


void Result::Clear() {
// @@protoc_insertion_point(message_clear_start:Persistance.Result)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  ::memset(&light_intensity_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&is_converged_) -
      reinterpret_cast<char*>(&light_intensity_)) + sizeof(is_converged_));
  _internal_metadata_.Clear();
}

bool Result::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:Persistance.Result)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // float light_intensity = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(13u /* 13 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   float, ::google::protobuf::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &light_intensity_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // bool is_converged = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(16u /* 16 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   bool, ::google::protobuf::internal::WireFormatLite::TYPE_BOOL>(
                 input, &is_converged_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:Persistance.Result)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:Persistance.Result)
  return false;
#undef DO_
}

void Result::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:Persistance.Result)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // float light_intensity = 1;
  if (this->light_intensity() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteFloat(1, this->light_intensity(), output);
  }

  // bool is_converged = 2;
  if (this->is_converged() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteBool(2, this->is_converged(), output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:Persistance.Result)
}

::google::protobuf::uint8* Result::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:Persistance.Result)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // float light_intensity = 1;
  if (this->light_intensity() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteFloatToArray(1, this->light_intensity(), target);
  }

  // bool is_converged = 2;
  if (this->is_converged() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteBoolToArray(2, this->is_converged(), target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:Persistance.Result)
  return target;
}

size_t Result::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:Persistance.Result)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // float light_intensity = 1;
  if (this->light_intensity() != 0) {
    total_size += 1 + 4;
  }

  // bool is_converged = 2;
  if (this->is_converged() != 0) {
    total_size += 1 + 1;
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void Result::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:Persistance.Result)
  GOOGLE_DCHECK_NE(&from, this);
  const Result* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const Result>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:Persistance.Result)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:Persistance.Result)
    MergeFrom(*source);
  }
}

void Result::MergeFrom(const Result& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:Persistance.Result)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.light_intensity() != 0) {
    set_light_intensity(from.light_intensity());
  }
  if (from.is_converged() != 0) {
    set_is_converged(from.is_converged());
  }
}

void Result::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:Persistance.Result)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Result::CopyFrom(const Result& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:Persistance.Result)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Result::IsInitialized() const {
  return true;
}

void Result::Swap(Result* other) {
  if (other == this) return;
  InternalSwap(other);
}
void Result::InternalSwap(Result* other) {
  using std::swap;
  swap(light_intensity_, other->light_intensity_);
  swap(is_converged_, other->is_converged_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata Result::GetMetadata() const {
  protobuf_Result_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_Result_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace Persistance
namespace google {
namespace protobuf {
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::Persistance::Result* Arena::CreateMaybeMessage< ::Persistance::Result >(Arena* arena) {
  return Arena::CreateInternal< ::Persistance::Result >(arena);
}
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
