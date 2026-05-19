//! Benchmarks for the UOR term language lexer and parser.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hologram_compiler::term_parser;

/// Generate a nested expression of depth `n`: `add(neg(succ(...42...)), 1)`
fn generate_nested_expr(depth: usize) -> String {
    let mut s = String::with_capacity(depth * 8);
    for i in 0..depth {
        match i % 3 {
            0 => s.push_str("add("),
            1 => s.push_str("neg("),
            _ => s.push_str("succ("),
        }
    }
    s.push_str("42");
    for i in 0..depth {
        if i % 3 == 0 {
            s.push_str(", 1)");
        } else {
            s.push(')');
        }
    }
    s
}

fn bench_parse_10(c: &mut Criterion) {
    let src = generate_nested_expr(10);
    c.bench_function("parse/10_nodes", |b| {
        b.iter(|| term_parser::parse(black_box(&src)).unwrap())
    });
}

fn bench_parse_100(c: &mut Criterion) {
    let src = generate_nested_expr(100);
    c.bench_function("parse/100_nodes", |b| {
        b.iter(|| term_parser::parse(black_box(&src)).unwrap())
    });
}

fn bench_parse_1000(c: &mut Criterion) {
    let src = generate_nested_expr(1000);
    c.bench_function("parse/1000_nodes", |b| {
        b.iter(|| term_parser::parse(black_box(&src)).unwrap())
    });
}

/// Benchmark lexer only (tokenize without building AST).
fn bench_lex_1000(c: &mut Criterion) {
    let src = generate_nested_expr(1000);
    c.bench_function("lex/1000_nodes", |b| {
        b.iter(|| {
            let mut lexer = term_parser::lexer::Lexer::new(black_box(&src));
            loop {
                let tok = lexer.next_token().unwrap();
                if tok == term_parser::lexer::Token::Eof {
                    break;
                }
            }
        })
    });
}

criterion_group!(
    benches,
    bench_parse_10,
    bench_parse_100,
    bench_parse_1000,
    bench_lex_1000
);
criterion_main!(benches);
