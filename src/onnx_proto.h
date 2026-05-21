#pragma once
// Minimal ONNX protobuf reader — no protobuf library dependency.
// Reads just enough of the ONNX format to extract graph topology,
// node attributes, tensor shapes/types, and initializer data.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>

namespace onnx_proto {

// Protobuf wire types
enum WireType : uint8_t {
    VARINT = 0,
    FIXED64 = 1,
    LENGTH_DELIMITED = 2,
    FIXED32 = 5
};

// ONNX TensorProto.DataType enum
enum DataType : int {
    UNDEFINED = 0, FLOAT = 1, UINT8 = 2, INT8 = 3,
    UINT16 = 4, INT16 = 5, INT32 = 6, INT64 = 7,
    STRING = 8, BOOL = 9, FLOAT16 = 10, DOUBLE = 11,
    UINT32 = 12, UINT64 = 13
};

// ONNX AttributeProto.AttributeType enum
enum AttrType : int {
    ATTR_UNDEFINED = 0, ATTR_FLOAT = 1, ATTR_INT = 2, ATTR_STRING = 3,
    ATTR_TENSOR = 4, ATTR_GRAPH = 5,
    ATTR_FLOATS = 6, ATTR_INTS = 7, ATTR_STRINGS = 8,
    ATTR_TENSORS = 9, ATTR_GRAPHS = 10
};

// ---- Protobuf decoding primitives ----

struct Reader {
    const uint8_t* data;
    const uint8_t* end;

    Reader(const uint8_t* d, size_t len) : data(d), end(d + len) {}

    bool has_more() const { return data < end; }

    uint64_t read_varint() {
        uint64_t result = 0;
        int shift = 0;
        while (data < end) {
            uint8_t b = *data++;
            result |= (uint64_t)(b & 0x7F) << shift;
            if (!(b & 0x80)) return result;
            shift += 7;
        }
        throw std::runtime_error("Truncated varint");
    }

    uint32_t read_fixed32() {
        if (data + 4 > end) throw std::runtime_error("Truncated fixed32");
        uint32_t v;
        memcpy(&v, data, 4);
        data += 4;
        return v;
    }

    uint64_t read_fixed64() {
        if (data + 8 > end) throw std::runtime_error("Truncated fixed64");
        uint64_t v;
        memcpy(&v, data, 8);
        data += 8;
        return v;
    }

    Reader read_length_delimited() {
        uint64_t len = read_varint();
        if (data + len > end) throw std::runtime_error("Truncated length-delimited field");
        Reader sub(data, len);
        data += len;
        return sub;
    }

    std::string read_string() {
        uint64_t len = read_varint();
        if (data + len > end) throw std::runtime_error("Truncated string");
        std::string s(reinterpret_cast<const char*>(data), len);
        data += len;
        return s;
    }

    void skip_field(uint8_t wire_type) {
        switch (wire_type) {
            case VARINT: read_varint(); break;
            case FIXED64: data += 8; break;
            case LENGTH_DELIMITED: {
                auto len = read_varint();
                if (data + len > end) {
                    char msg[128];
                    snprintf(msg, sizeof(msg),
                        "Truncated length-delimited skip: len=%llu remaining=%td",
                        (unsigned long long)len, end - data);
                    throw std::runtime_error(msg);
                }
                data += len;
                break;
            }
            case 3: // Start group (deprecated) — skip until end group
                while (has_more()) {
                    auto [f, w] = read_tag();
                    if (w == 4) break; // End group
                    skip_field(w);
                }
                break;
            case 4: break; // End group — should not appear standalone
            case FIXED32: data += 4; break;
            default: {
                char msg[64];
                snprintf(msg, sizeof(msg),
                    "Unknown wire type %d at offset %td", wire_type, data - end);
                throw std::runtime_error(msg);
            }
        }
    }

    // Read field tag: returns (field_number, wire_type)
    std::pair<uint32_t, uint8_t> read_tag() {
        uint64_t v = read_varint();
        return {static_cast<uint32_t>(v >> 3), static_cast<uint8_t>(v & 0x7)};
    }
};

// ---- ONNX data structures ----

struct Dimension {
    bool is_symbolic = false;
    int64_t value = -1;       // -1 if symbolic/unknown
    std::string param;        // symbolic name if is_symbolic
};

struct TypeInfo {
    DataType elem_type = UNDEFINED;
    std::vector<Dimension> shape;
};

struct Attribute {
    std::string name;
    AttrType type = ATTR_UNDEFINED;
    float f = 0;
    int64_t i = 0;
    std::string s;
    std::vector<float> floats;
    std::vector<int64_t> ints;
    std::vector<std::string> strings;
};

struct NodeProto {
    std::string op_type;
    std::string name;
    std::string domain;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<Attribute> attributes;
};

struct TensorProto {
    std::string name;
    DataType data_type = UNDEFINED;
    std::vector<int64_t> dims;
    // Raw data storage
    std::vector<uint8_t> raw_data;
    std::vector<float> float_data;
    std::vector<double> double_data;
    std::vector<int32_t> int32_data;
    std::vector<int64_t> int64_data;
    // External data
    std::string external_location;
    int64_t external_offset = 0;
    int64_t external_length = 0;
};

struct ValueInfoProto {
    std::string name;
    TypeInfo type;
};

struct GraphProto {
    std::string name;
    std::vector<NodeProto> nodes;
    std::vector<ValueInfoProto> inputs;
    std::vector<ValueInfoProto> outputs;
    std::vector<TensorProto> initializers;
};

struct ModelProto {
    int64_t ir_version = 0;
    int64_t model_version = 0;
    std::string producer_name;
    GraphProto graph;
};

// ---- Parsing functions ----

inline TypeInfo parse_type_proto(Reader r) {
    TypeInfo ti;
    // TypeProto: field 1 = tensor_type (message TensorTypeProto)
    while (r.has_more()) {
        auto [field, wire] = r.read_tag();
        if (field == 1 && wire == LENGTH_DELIMITED) {
            // TensorTypeProto
            auto sub = r.read_length_delimited();
            while (sub.has_more()) {
                auto [tf, tw] = sub.read_tag();
                if (tf == 1 && tw == VARINT) {
                    ti.elem_type = static_cast<DataType>(sub.read_varint());
                } else if (tf == 2 && tw == LENGTH_DELIMITED) {
                    // TensorShapeProto
                    auto shape_r = sub.read_length_delimited();
                    while (shape_r.has_more()) {
                        auto [sf, sw] = shape_r.read_tag();
                        if (sf == 1 && sw == LENGTH_DELIMITED) {
                            // Dimension
                            auto dim_r = shape_r.read_length_delimited();
                            Dimension dim;
                            while (dim_r.has_more()) {
                                auto [df, dw] = dim_r.read_tag();
                                if (df == 1 && dw == VARINT) {
                                    dim.value = dim_r.read_varint();
                                    dim.is_symbolic = false;
                                } else if (df == 2 && dw == LENGTH_DELIMITED) {
                                    dim.param = dim_r.read_string();
                                    // Oops: read_string already consumed length
                                    // Actually read_length_delimited gives us a sub-reader
                                    // Let me fix: field 2 in Dimension is dim_param (string)
                                    dim.is_symbolic = true;
                                    dim.value = -1;
                                } else {
                                    dim_r.skip_field(dw);
                                }
                            }
                            ti.shape.push_back(dim);
                        } else {
                            shape_r.skip_field(sw);
                        }
                    }
                } else {
                    sub.skip_field(tw);
                }
            }
        } else {
            r.skip_field(wire);
        }
    }
    return ti;
}

inline ValueInfoProto parse_value_info(Reader r) {
    ValueInfoProto vi;
    while (r.has_more()) {
        auto [field, wire] = r.read_tag();
        if (field == 1 && wire == LENGTH_DELIMITED) {
            vi.name = r.read_string();
        } else if (field == 2 && wire == LENGTH_DELIMITED) {
            vi.type = parse_type_proto(r.read_length_delimited());
        } else {
            r.skip_field(wire);
        }
    }
    return vi;
}

inline void parse_external_data(Reader r, TensorProto& tp) {
    // StringStringEntryProto: field 1 = key, field 2 = value
    std::string key, value;
    while (r.has_more()) {
        auto [field, wire] = r.read_tag();
        if (field == 1 && wire == LENGTH_DELIMITED) {
            key = r.read_string();
        } else if (field == 2 && wire == LENGTH_DELIMITED) {
            value = r.read_string();
        } else {
            r.skip_field(wire);
        }
    }
    if (key == "location") tp.external_location = value;
    else if (key == "offset") tp.external_offset = std::stoll(value);
    else if (key == "length") tp.external_length = std::stoll(value);
}

inline TensorProto parse_tensor(Reader r) {
    // TensorProto field numbers (from onnx.proto3):
    //  1: dims (repeated int64)
    //  2: data_type (int32)
    //  3: segment (deprecated)
    //  4: float_data (repeated float, packed)
    //  5: int32_data (repeated int32, packed)
    //  6: string_data (repeated bytes)
    //  7: int64_data (repeated int64, packed)
    //  8: name (string)
    //  9: raw_data (bytes)
    // 10: double_data (repeated double, packed)
    // 11: uint64_data (repeated uint64, packed)
    // 12: doc_string (string)
    // 13: external_data (repeated StringStringEntryProto)
    // 14: data_location (DataLocation enum)
    TensorProto tp;
    while (r.has_more()) {
        auto [field, wire] = r.read_tag();
        if (field == 1) { // dims
            if (wire == VARINT) tp.dims.push_back(r.read_varint());
            else if (wire == LENGTH_DELIMITED) {
                auto sub = r.read_length_delimited();
                while (sub.has_more()) tp.dims.push_back(sub.read_varint());
            } else r.skip_field(wire);
        } else if (field == 2 && wire == VARINT) {
            tp.data_type = static_cast<DataType>(r.read_varint());
        } else if (field == 4 && wire == LENGTH_DELIMITED) {
            auto sub = r.read_length_delimited();
            while (sub.has_more()) {
                float v; uint32_t bits = sub.read_fixed32();
                memcpy(&v, &bits, 4); tp.float_data.push_back(v);
            }
        } else if (field == 5 && wire == LENGTH_DELIMITED) {
            auto sub = r.read_length_delimited();
            while (sub.has_more()) tp.int32_data.push_back(static_cast<int32_t>(sub.read_varint()));
        } else if (field == 7 && wire == LENGTH_DELIMITED) {
            auto sub = r.read_length_delimited();
            while (sub.has_more()) tp.int64_data.push_back(static_cast<int64_t>(sub.read_varint()));
        } else if (field == 8 && wire == LENGTH_DELIMITED) {
            tp.name = r.read_string();
        } else if (field == 9 && wire == LENGTH_DELIMITED) {
            auto sub = r.read_length_delimited();
            tp.raw_data.assign(sub.data, sub.end);
        } else if (field == 10 && wire == LENGTH_DELIMITED) {
            auto sub = r.read_length_delimited();
            while (sub.has_more()) {
                double v; uint64_t bits = sub.read_fixed64();
                memcpy(&v, &bits, 8); tp.double_data.push_back(v);
            }
        } else if (field == 13 && wire == LENGTH_DELIMITED) {
            parse_external_data(r.read_length_delimited(), tp);
        } else if (field == 14 && wire == VARINT) {
            r.read_varint(); // data_location, skip
        } else {
            r.skip_field(wire);
        }
    }
    return tp;
}

inline Attribute parse_attribute(Reader r) {
    // AttributeProto field numbers (from onnx.proto3):
    //  1: name (string)
    //  2: f (float)
    //  3: i (int64)
    //  4: s (bytes)
    //  5: t (TensorProto)
    //  6: g (GraphProto)
    //  7: floats (repeated float)
    //  8: ints (repeated int64)
    //  9: strings (repeated bytes)
    // 10: tensors (repeated TensorProto)
    // 11: graphs (repeated GraphProto)
    // 13: doc_string (string)
    // 20: type (AttributeType)
    // 21: ref_attr_name (string)
    Attribute attr;
    while (r.has_more()) {
        auto [field, wire] = r.read_tag();
        if (field == 1 && wire == LENGTH_DELIMITED) {
            attr.name = r.read_string();
        } else if (field == 2 && wire == FIXED32) {
            uint32_t bits = r.read_fixed32(); memcpy(&attr.f, &bits, 4);
        } else if (field == 3 && wire == VARINT) {
            attr.i = static_cast<int64_t>(r.read_varint());
        } else if (field == 4 && wire == LENGTH_DELIMITED) {
            attr.s = r.read_string();
        } else if (field == 7) { // floats
            if (wire == LENGTH_DELIMITED) {
                auto sub = r.read_length_delimited();
                while (sub.has_more()) {
                    float v; uint32_t bits = sub.read_fixed32();
                    memcpy(&v, &bits, 4); attr.floats.push_back(v);
                }
            } else if (wire == FIXED32) {
                float v; uint32_t bits = r.read_fixed32();
                memcpy(&v, &bits, 4); attr.floats.push_back(v);
            } else r.skip_field(wire);
        } else if (field == 8) { // ints
            if (wire == LENGTH_DELIMITED) {
                auto sub = r.read_length_delimited();
                while (sub.has_more()) attr.ints.push_back(static_cast<int64_t>(sub.read_varint()));
            } else if (wire == VARINT) {
                attr.ints.push_back(static_cast<int64_t>(r.read_varint()));
            } else r.skip_field(wire);
        } else if (field == 9 && wire == LENGTH_DELIMITED) {
            attr.strings.push_back(r.read_string());
        } else if (field == 20 && wire == VARINT) {
            attr.type = static_cast<AttrType>(r.read_varint());
        } else {
            r.skip_field(wire);
        }
    }
    return attr;
}

inline NodeProto parse_node(Reader r) {
    // NodeProto field numbers:
    //  1: input (repeated string)
    //  2: output (repeated string)
    //  3: name (string)
    //  4: op_type (string)
    //  5: attribute (repeated AttributeProto)
    //  7: domain (string)
    //  8: overload (string)
    // 13: device_configurations (repeated)
    NodeProto node;
    while (r.has_more()) {
        auto [field, wire] = r.read_tag();
        if (wire == LENGTH_DELIMITED) {
            switch (field) {
                case 1: node.inputs.push_back(r.read_string()); break;
                case 2: node.outputs.push_back(r.read_string()); break;
                case 3: node.name = r.read_string(); break;
                case 4: node.op_type = r.read_string(); break;
                case 5: node.attributes.push_back(parse_attribute(r.read_length_delimited())); break;
                case 7: node.domain = r.read_string(); break;
                default: r.skip_field(wire); break;
            }
        } else {
            r.skip_field(wire);
        }
    }
    return node;
}

inline GraphProto parse_graph(Reader r) {
    // GraphProto field numbers from onnx.proto3:
    //  1: node (repeated NodeProto)
    //  2: name (string)
    //  5: initializer (repeated TensorProto)
    // 10: doc_string (string)
    // 11: input (repeated ValueInfoProto)
    // 12: output (repeated ValueInfoProto)
    // 13: value_info (repeated ValueInfoProto)
    // 14: quantization_annotation
    // 15: sparse_initializer
    GraphProto graph;
    while (r.has_more()) {
        auto [field, wire] = r.read_tag();
        if (wire == LENGTH_DELIMITED) {
            switch (field) {
                case 1:
                    try {
                        graph.nodes.push_back(parse_node(r.read_length_delimited()));
                    } catch (const std::exception& e) {
                        char msg[256];
                        snprintf(msg, sizeof(msg), "Error parsing node %zu: %s",
                            graph.nodes.size(), e.what());
                        throw std::runtime_error(msg);
                    }
                    break;
                case 2:  graph.name = r.read_string(); break;
                case 5:
                    try {
                        graph.initializers.push_back(parse_tensor(r.read_length_delimited()));
                    } catch (const std::exception& e) {
                        char msg[256];
                        snprintf(msg, sizeof(msg), "Error parsing initializer %zu: %s",
                            graph.initializers.size(), e.what());
                        throw std::runtime_error(msg);
                    }
                    break;
                case 10: r.skip_field(wire); break; // doc_string
                case 11: graph.inputs.push_back(parse_value_info(r.read_length_delimited())); break;
                case 12: graph.outputs.push_back(parse_value_info(r.read_length_delimited())); break;
                case 13: r.skip_field(wire); break; // value_info
                case 14: r.skip_field(wire); break; // quantization_annotation
                case 15: r.skip_field(wire); break; // sparse_initializer
                default: r.skip_field(wire); break;
            }
        } else if (wire == VARINT) {
            r.read_varint(); // skip any varint fields
        } else {
            r.skip_field(wire);
        }
    }
    return graph;
}

inline ModelProto parse_model(Reader r) {
    // ModelProto field numbers:
    //  1: ir_version (int64)
    //  2: producer_name (string)
    //  3: producer_version (string)
    //  4: domain (string)
    //  5: model_version (int64)
    //  6: doc_string (string)
    //  7: graph (GraphProto)
    //  8: opset_import (repeated OperatorSetIdProto)
    // All string/message fields use LENGTH_DELIMITED wire type.
    ModelProto model;
    while (r.has_more()) {
        auto [field, wire] = r.read_tag();
        if (wire == VARINT) {
            if (field == 1) model.ir_version = r.read_varint();
            else if (field == 5) model.model_version = r.read_varint();
            else r.read_varint(); // skip unknown varint
        } else if (wire == LENGTH_DELIMITED) {
            if (field == 2) model.producer_name = r.read_string();
            else if (field == 7) model.graph = parse_graph(r.read_length_delimited());
            else r.skip_field(wire); // skip other length-delimited fields
        } else {
            r.skip_field(wire);
        }
    }
    return model;
}

// Read and parse an ONNX file
inline ModelProto load(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("Cannot open file: " + path);
    size_t size = ifs.tellg();
    ifs.seekg(0);
    std::vector<uint8_t> buf(size);
    ifs.read(reinterpret_cast<char*>(buf.data()), size);
    return parse_model(Reader(buf.data(), buf.size()));
}

// Load external data for a tensor
inline void load_external_data(TensorProto& tp, const std::string& model_dir) {
    if (tp.external_location.empty()) return;
    std::string path = model_dir + "/" + tp.external_location;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open external data: " + path);
    if (tp.external_offset > 0) ifs.seekg(tp.external_offset);
    size_t len = tp.external_length > 0 ? tp.external_length : 0;
    if (len == 0) {
        ifs.seekg(0, std::ios::end);
        len = static_cast<size_t>(ifs.tellg()) - tp.external_offset;
        ifs.seekg(tp.external_offset);
    }
    tp.raw_data.resize(len);
    ifs.read(reinterpret_cast<char*>(tp.raw_data.data()), len);
}

} // namespace onnx_proto
