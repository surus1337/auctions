use leptos::*;
use rand::Rng;
use chrono;
use plotters::prelude::*;
use plotters::series::LineSeries;
use plotters_canvas::CanvasBackend;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;
use std::collections::HashMap;

// Helper function for setTimeout - fixed to use FnMut
fn set_timeout(f: impl FnMut() + 'static, duration: std::time::Duration) {
    let callback = wasm_bindgen::closure::Closure::wrap(Box::new(f) as Box<dyn FnMut()>);
    web_sys::window()
        .unwrap()
        .set_timeout_with_callback_and_timeout_and_arguments_0(
            callback.as_ref().unchecked_ref(),
            duration.as_millis() as i32,
        )
        .unwrap();
    callback.forget();
}

// Scenario templates
#[derive(Debug, Clone)]
struct ScenarioTemplate {
    name: &'static str,
    trust_assets: u32,
    redemption_percentage: u32,
    num_bidders: usize,
    total_tokens: usize,
    trust_assets_growth: u32,
    redemption_percentage_growth: u32,
    bidder_growth: u32,
    token_growth: u32,
}

const SCENARIO_TEMPLATES: &[ScenarioTemplate] = &[
    ScenarioTemplate {
        name: "Default",
        trust_assets: 350000,
        redemption_percentage: 5,
        num_bidders: 30,
        total_tokens: 60000,
        trust_assets_growth: 30,
        redemption_percentage_growth: 20,
        bidder_growth: 30,
        token_growth: 40,
    },
    ScenarioTemplate {
        name: "Market Crash",
        trust_assets: 30000,
        redemption_percentage: 15,
        num_bidders: 8,
        total_tokens: 80,
        trust_assets_growth: 10,
        redemption_percentage_growth: 5,
        bidder_growth: 50,
        token_growth: 60,
    },
    ScenarioTemplate {
        name: "New Entrant Wave",
        trust_assets: 75000,
        redemption_percentage: 25,
        num_bidders: 12,
        total_tokens: 40,
        trust_assets_growth: 40,
        redemption_percentage_growth: 30,
        bidder_growth: 80,
        token_growth: 100,
    },
    ScenarioTemplate {
        name: "Mature Market",
        trust_assets: 200000,
        redemption_percentage: 10,
        num_bidders: 20,
        total_tokens: 200,
        trust_assets_growth: 15,
        redemption_percentage_growth: 10,
        bidder_growth: 15,
        token_growth: 20,
    },
];

// Active tab enum
#[derive(Debug, Clone, Copy, PartialEq)]
enum ActiveTab {
    Simulation,
    Analytics,
    BidDetails,
    Configuration,
}

// Market conditions that affect bidding behavior
#[derive(Debug, Clone, Copy, PartialEq)]
enum MarketCondition {
    Bull,
    Neutral,
    Bear,
    Volatile,
}

impl MarketCondition {
    fn name(&self) -> &str {
        match self {
            MarketCondition::Bull => "Bull Market",
            MarketCondition::Neutral => "Neutral Market",
            MarketCondition::Bear => "Bear Market",
            MarketCondition::Volatile => "Volatile Market",
        }
    }
    
    fn description(&self) -> &str {
        match self {
            MarketCondition::Bull => "High optimism, inflated prices",
            MarketCondition::Neutral => "Normal market conditions",
            MarketCondition::Bear => "Low confidence, depressed prices",
            MarketCondition::Volatile => "Unpredictable price swings",
        }
    }
    
    fn multiplier(&self) -> f64 {
        match self {
            MarketCondition::Bull => 1.3,
            MarketCondition::Neutral => 1.0,
            MarketCondition::Bear => 0.7,
            MarketCondition::Volatile => {
                let mut rng = rand::thread_rng();
                rng.gen_range(0.5..1.5)
            }
        }
    }
}

impl std::fmt::Display for MarketCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BidderProfile {
    VeryConservative,
    Conservative,
    Moderate,
    Aggressive,
    VeryAggressive,
}

impl BidderProfile {
    fn price_range(&self) -> (u32, u32) {
        match self {
            BidderProfile::VeryConservative => (700, 1200),
            BidderProfile::Conservative => (500, 1000),
            BidderProfile::Moderate => (300, 800),
            BidderProfile::Aggressive => (200, 600),
            BidderProfile::VeryAggressive => (100, 400),
        }
    }

    fn description(&self) -> &str {
        match self {
            BidderProfile::VeryConservative => "Asks for high prices, unwilling to sell for less",
            BidderProfile::Conservative => "Asks for above-average prices",
            BidderProfile::Moderate => "Asks for average market prices",
            BidderProfile::Aggressive => "Willing to sell for below-average prices",
            BidderProfile::VeryAggressive => "Willing to sell for very low prices",
        }
    }
    
    fn hold_strategy(&self) -> f64 {
        match self {
            BidderProfile::VeryConservative => 0.8,  // Holds 80% of tokens
            BidderProfile::Conservative => 0.6,       // Holds 60%
            BidderProfile::Moderate => 0.4,           // Holds 40%
            BidderProfile::Aggressive => 0.2,         // Holds 20%
            BidderProfile::VeryAggressive => 0.1,     // Holds 10%
        }
    }
    
    fn learning_rate(&self) -> f64 {
        match self {
            BidderProfile::VeryConservative => 0.1,  // Learns slowly
            BidderProfile::Conservative => 0.15,
            BidderProfile::Moderate => 0.2,
            BidderProfile::Aggressive => 0.25,
            BidderProfile::VeryAggressive => 0.3,     // Learns quickly
        }
    }

    fn color_class(&self) -> &str {
        match self {
            BidderProfile::VeryConservative => "bg-blue-800",
            BidderProfile::Conservative => "bg-blue-500",
            BidderProfile::Moderate => "bg-purple-500",
            BidderProfile::Aggressive => "bg-orange-500",
            BidderProfile::VeryAggressive => "bg-red-500",
        }
    }
}

impl std::fmt::Display for BidderProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            BidderProfile::VeryConservative => "Very Conservative",
            BidderProfile::Conservative => "Conservative",
            BidderProfile::Moderate => "Moderate",
            BidderProfile::Aggressive => "Aggressive",
            BidderProfile::VeryAggressive => "Very Aggressive",
        };
        write!(f, "{}", name)
    }
}

#[derive(Debug, Clone)]
struct Bid {
    token_id: usize,
    price: u32,
}

#[derive(Debug, Clone)]
struct Bidder {
    id: usize,
    name: String,
    tokens: usize,
    profile: BidderProfile,
    bids: Vec<Bid>,
    // Enhanced fields for learning
    success_rate: f64,
    total_revenue: u32,
    auction_history: Vec<BidderAuctionHistory>,
    adapted_price_range: (u32, u32),  // Learned price range
    new_tokens_received: usize,        // Tokens received in current round
}

#[derive(Debug, Clone)]
struct BidderAuctionHistory {
    auction_number: usize,
    tokens_offered: usize,
    tokens_sold: usize,
    revenue: u32,
    average_sale_price: f64,
}

#[derive(Debug, Clone)]
struct ClearedBid {
    bidder_id: usize,
    bidder_name: String,
    bidder_profile: BidderProfile,
    token_id: usize,
    price: u32,
}

#[derive(Debug, Clone)]
struct BidderResult {
    id: usize,
    name: String,
    profile: BidderProfile,
    initial_tokens: usize,
    tokens_sold: usize,
    tokens_left: usize,
    tokens_held: usize,    // Tokens intentionally held back
    revenue: u32,
    new_tokens: usize,     // New tokens received this round
    success_rate: f64,     // Current success rate
}

#[derive(Debug, Clone)]
struct AuctionResults {
    auction_number: usize,
    trust_assets: u32,
    redemption_pool: u32,
    redemption_percentage: u32,
    market_condition: MarketCondition,
    total_bids: usize,
    cleared_bids: Vec<ClearedBid>,
    rejected_bids: Vec<ClearedBid>,
    remaining_pool: u32,
    total_spent: u32,
    tokens_purchased: usize,
    average_purchase_price: f64,
    bidder_results: Vec<BidderResult>,
    timestamp: String,
}

fn generate_initial_bidders(num_bidders: usize, total_tokens: usize, _use_profiles: bool) -> Vec<Bidder> {
    let mut rng = rand::thread_rng();
    let mut remaining_tokens = total_tokens;
    let mut bidders = Vec::new();

    let profiles = [
        BidderProfile::VeryConservative,
        BidderProfile::Conservative,
        BidderProfile::Moderate,
        BidderProfile::Aggressive,
        BidderProfile::VeryAggressive,
    ];

    for i in 0..num_bidders {
        let tokens = if i == num_bidders - 1 {
            remaining_tokens
        } else {
            let max_tokens = (total_tokens / num_bidders * 2).min(remaining_tokens);
            if max_tokens > 0 {
                rng.gen_range(1..=max_tokens)
            } else {
                0
            }
        };
        remaining_tokens = remaining_tokens.saturating_sub(tokens);

        let profile_index = rng.gen_range(0..profiles.len());
        let profile = profiles[profile_index];
        let price_range = profile.price_range();

        let bidder = Bidder {
            id: i + 1,
            name: format!("Bidder {}", i + 1),
            tokens,
            profile,
            bids: Vec::new(),
            success_rate: 0.0,
            total_revenue: 0,
            auction_history: Vec::new(),
            adapted_price_range: price_range,
            new_tokens_received: 0,
        };

        bidders.push(bidder);
    }

    bidders
}

fn generate_bids_for_bidders(
    bidders: &mut Vec<Bidder>, 
    use_profiles: bool, 
    market_condition: MarketCondition,
    auction_number: usize
) {
    let mut rng = rand::thread_rng();
    
    for bidder in bidders.iter_mut() {
        bidder.bids.clear();
        
        if bidder.tokens == 0 {
            continue;
        }
        
        // Calculate how many tokens to hold based on profile
        let hold_ratio = if use_profiles { 
            bidder.profile.hold_strategy() 
        } else { 
            0.3 // Default 30% hold
        };
        
        // Adjust hold ratio based on past success
        let adjusted_hold_ratio = if bidder.auction_history.len() >= 2 {
            if bidder.success_rate < 30.0 {
                // If struggling to sell, hold less
                (hold_ratio * 0.7).max(0.1)
            } else if bidder.success_rate > 70.0 {
                // If very successful, can afford to hold more
                (hold_ratio * 1.2).min(0.9)
            } else {
                hold_ratio
            }
        } else {
            hold_ratio
        };
        
        let tokens_to_hold = (bidder.tokens as f64 * adjusted_hold_ratio) as usize;
        let tokens_to_sell = bidder.tokens - tokens_to_hold;
        
        // Generate bids for tokens to sell
        for j in 0..tokens_to_sell {
            let price = if use_profiles {
                let (base_min, base_max) = bidder.adapted_price_range;
                let base_price = rng.gen_range(base_min..=base_max);
                
                // Apply market conditions
                let market_multiplier = market_condition.multiplier();
                let market_adjusted = (base_price as f64 * market_multiplier) as u32;
                
                // Apply auction number factor (prices tend to rise over time)
                let auction_factor = 1.0 + (auction_number as f64 - 1.0) * 0.05;
                (market_adjusted as f64 * auction_factor) as u32
            } else {
                let base_price = rng.gen_range(100..=1000);
                let market_multiplier = market_condition.multiplier();
                (base_price as f64 * market_multiplier) as u32
            };

            bidder.bids.push(Bid {
                token_id: j + 1,
                price,
            });
        }
    }
}

fn run_auction_with_learning(
    bidders: &mut Vec<Bidder>, 
    trust_pool: u32, 
    auction_number: usize,
    trust_assets: u32,
    redemption_percentage: u32,
    market_condition: MarketCondition
) -> AuctionResults {
    let mut all_bids: Vec<ClearedBid> = Vec::new();
    
    // Track tokens offered per bidder
    let mut tokens_offered: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    
    for bidder in bidders.iter() {
        tokens_offered.insert(bidder.id, bidder.bids.len());
        
        for bid in &bidder.bids {
            all_bids.push(ClearedBid {
                bidder_id: bidder.id,
                bidder_name: bidder.name.clone(),
                bidder_profile: bidder.profile,
                token_id: bid.token_id,
                price: bid.price,
            });
        }
    }

    all_bids.sort_by(|a, b| a.price.cmp(&b.price));

    let mut remaining_pool = trust_pool;
    let mut cleared_bids = Vec::new();
    let mut rejected_bids = Vec::new();

    for bid in all_bids.iter() {
        if remaining_pool >= bid.price {
            cleared_bids.push(bid.clone());
            remaining_pool -= bid.price;
        } else {
            rejected_bids.push(bid.clone());
        }
    }

    // Calculate results and update bidder learning
    let mut bidder_results = Vec::new();
    
    for bidder in bidders.iter_mut() {
        let bidder_cleared_bids: Vec<&ClearedBid> = cleared_bids
            .iter()
            .filter(|bid| bid.bidder_id == bidder.id)
            .collect();
        
        let revenue: u32 = bidder_cleared_bids
            .iter()
            .map(|bid| bid.price)
            .sum();
        
        let tokens_sold = bidder_cleared_bids.len();
        let tokens_offered_count = *tokens_offered.get(&bidder.id).unwrap_or(&0);
        let tokens_held = bidder.tokens - tokens_offered_count;
        
        // Calculate success rate for this auction
        let current_success_rate = if tokens_offered_count > 0 {
            (tokens_sold as f64 / tokens_offered_count as f64) * 100.0
        } else {
            0.0
        };
        
        // Update bidder's history
        if tokens_offered_count > 0 {
            let avg_sale_price = if tokens_sold > 0 {
                revenue as f64 / tokens_sold as f64
            } else {
                0.0
            };
            
            bidder.auction_history.push(BidderAuctionHistory {
                auction_number,
                tokens_offered: tokens_offered_count,
                tokens_sold,
                revenue,
                average_sale_price: avg_sale_price,
            });
        }
        
        // Update total revenue
        bidder.total_revenue += revenue;
        
        // Calculate overall success rate
        let total_offered: usize = bidder.auction_history.iter().map(|h| h.tokens_offered).sum();
        let total_sold: usize = bidder.auction_history.iter().map(|h| h.tokens_sold).sum();
        bidder.success_rate = if total_offered > 0 {
            (total_sold as f64 / total_offered as f64) * 100.0
        } else {
            0.0
        };
        
        // Adapt price range based on performance
        if bidder.auction_history.len() >= 2 {
            let learning_rate = bidder.profile.learning_rate();
            let (mut min_price, mut max_price) = bidder.adapted_price_range;
            
            if current_success_rate < 30.0 {
                // Lower prices if struggling
                min_price = (min_price as f64 * (1.0 - learning_rate * 0.5)).max(50.0) as u32;
                max_price = (max_price as f64 * (1.0 - learning_rate * 0.5)).max(100.0) as u32;
            } else if current_success_rate > 70.0 {
                // Raise prices if very successful
                min_price = (min_price as f64 * (1.0 + learning_rate * 0.3)).min(2000.0) as u32;
                max_price = (max_price as f64 * (1.0 + learning_rate * 0.3)).min(3000.0) as u32;
            }
            
            bidder.adapted_price_range = (min_price, max_price);
        }
        
        bidder_results.push(BidderResult {
            id: bidder.id,
            name: bidder.name.clone(),
            profile: bidder.profile,
            initial_tokens: bidder.tokens,
            tokens_sold,
            tokens_left: bidder.tokens - tokens_sold,
            tokens_held,
            revenue,
            new_tokens: bidder.new_tokens_received,
            success_rate: current_success_rate,
        });
        
        // Update tokens (remove sold ones)
        bidder.tokens -= tokens_sold;
        
        // Clear new tokens flag for next round
        bidder.new_tokens_received = 0;
    }

    let total_spent = trust_pool - remaining_pool;
    let tokens_purchased = cleared_bids.len();
    let average_purchase_price = if tokens_purchased > 0 {
        total_spent as f64 / tokens_purchased as f64
    } else {
        0.0
    };

    AuctionResults {
        auction_number,
        trust_assets,
        redemption_pool: trust_pool,
        redemption_percentage,
        market_condition,
        total_bids: all_bids.len(),
        cleared_bids,
        rejected_bids,
        remaining_pool,
        total_spent,
        tokens_purchased,
        average_purchase_price,
        bidder_results,
        timestamp: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
    }
}

fn distribute_new_tokens(
    bidders: &mut Vec<Bidder>,
    new_tokens: usize,
    num_new_bidders: usize,
) {
    let mut rng = rand::thread_rng();
    
    // Add new bidders if needed
    let current_bidder_count = bidders.len();
    if num_new_bidders > current_bidder_count {
        let profiles = [
            BidderProfile::VeryConservative,
            BidderProfile::Conservative,
            BidderProfile::Moderate,
            BidderProfile::Aggressive,
            BidderProfile::VeryAggressive,
        ];
        
        for i in current_bidder_count..num_new_bidders {
            let profile_index = rng.gen_range(0..profiles.len());
            let profile = profiles[profile_index];
            let price_range = profile.price_range();
            
            bidders.push(Bidder {
                id: i + 1,
                name: format!("Bidder {}", i + 1),
                tokens: 0,
                profile,
                bids: Vec::new(),
                success_rate: 0.0,
                total_revenue: 0,
                auction_history: Vec::new(),
                adapted_price_range: price_range,
                new_tokens_received: 0,
            });
        }
    }
    
    // Distribute new tokens
    if new_tokens > 0 {
        // 60% to new bidders, 40% to existing bidders
        let new_bidder_count = num_new_bidders.saturating_sub(current_bidder_count);
        let tokens_for_new = if new_bidder_count > 0 {
            (new_tokens as f64 * 0.6) as usize
        } else {
            0
        };
        let tokens_for_existing = new_tokens - tokens_for_new;
        
        // Distribute to new bidders
        if new_bidder_count > 0 && tokens_for_new > 0 {
            let tokens_per_new_bidder = tokens_for_new / new_bidder_count;
            let mut remaining = tokens_for_new % new_bidder_count;
            
            for bidder in bidders.iter_mut().skip(current_bidder_count) {
                let extra = if remaining > 0 { 
                    remaining -= 1;
                    1 
                } else { 
                    0 
                };
                let tokens_to_give = tokens_per_new_bidder + extra;
                bidder.tokens += tokens_to_give;
                bidder.new_tokens_received = tokens_to_give;
            }
        }
        
        // Distribute to existing bidders based on performance
        if tokens_for_existing > 0 && current_bidder_count > 0 {
            // Calculate total weight (inverse of success rate favors struggling bidders)
            let weights: Vec<f64> = bidders.iter()
                .take(current_bidder_count)
                .map(|b| {
                    if b.success_rate > 0.0 {
                        100.0 / b.success_rate  // Inverse weight
                    } else {
                        2.0  // Default weight for new bidders
                    }
                })
                .collect();
            
            let total_weight: f64 = weights.iter().sum();
            
            if total_weight > 0.0 {
                for (i, bidder) in bidders.iter_mut().take(current_bidder_count).enumerate() {
                    let share = weights[i] / total_weight;
                    let tokens_to_give = (tokens_for_existing as f64 * share) as usize;
                    bidder.tokens += tokens_to_give;
                    bidder.new_tokens_received += tokens_to_give;
                }
            }
        }
    }
}

// Chart drawing functions
fn draw_price_history_chart(
    canvas_id: &str,
    auction_history: &[AuctionResults],
) -> Result<(), Box<dyn std::error::Error>> {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .get_element_by_id(canvas_id)
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();
    
    let width = 800;
    let height = 400;
    canvas.set_width(width);
    canvas.set_height(height);
    
    let backend = CanvasBackend::new(canvas_id).expect("Cannot find canvas");
    let root = backend.into_drawing_area();
    root.fill(&WHITE)?;
    
    if auction_history.is_empty() {
        return Ok(());
    }
    
    let min_price = auction_history
        .iter()
        .map(|a| a.average_purchase_price as i32)
        .min()
        .unwrap_or(0);
    let max_price = auction_history
        .iter()
        .map(|a| a.average_purchase_price as i32)
        .max()
        .unwrap_or(100);
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Average Price History", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(
            1f32..auction_history.len() as f32,
            (min_price - 10)..(max_price + 10),
        )?;
    
    chart
        .configure_mesh()
        .x_desc("Auction Number")
        .y_desc("Average Price ($)")
        .draw()?;
    
    // Draw average price line
    chart
        .draw_series(LineSeries::new(
            auction_history
                .iter()
                .enumerate()
                .map(|(i, auction)| ((i + 1) as f32, auction.average_purchase_price as i32)),
            &BLUE,
        ))?
        .label("Average Price")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));
    
    // Draw pool utilization line
    chart
        .draw_series(LineSeries::new(
            auction_history
                .iter()
                .enumerate()
                .map(|(i, auction)| {
                    let utilization = (auction.total_spent as f32 / auction.redemption_pool as f32 * 100.0) as i32;
                    ((i + 1) as f32, utilization)
                }),
            &GREEN,
        ))?
        .label("Pool Utilization %")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &GREEN));
    
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    
    root.present()?;
    Ok(())
}

fn draw_trading_volume_chart(
    canvas_id: &str,
    auction_history: &[AuctionResults],
) -> Result<(), Box<dyn std::error::Error>> {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .get_element_by_id(canvas_id)
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();
    
    let width = 800;
    let height = 400;
    canvas.set_width(width);
    canvas.set_height(height);
    
    let backend = CanvasBackend::new(canvas_id).expect("Cannot find canvas");
    let root = backend.into_drawing_area();
    root.fill(&WHITE)?;
    
    if auction_history.is_empty() {
        return Ok(());
    }
    
    let max_tokens = auction_history
        .iter()
        .map(|a| a.tokens_purchased)
        .max()
        .unwrap_or(10) as i32;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Trading Volume", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(
            0f32..(auction_history.len() as f32 + 1.0),
            0..(max_tokens + 5),
        )?;
    
    chart
        .configure_mesh()
        .x_desc("Auction Number")
        .y_desc("Tokens Traded")
        .draw()?;
    
    chart.draw_series(
        auction_history
            .iter()
            .enumerate()
            .map(|(i, auction)| {
                Rectangle::new(
                    [((i as f32 + 0.5), 0), ((i as f32 + 1.5), auction.tokens_purchased as i32)],
                    BLUE.filled(),
                )
            }),
    )?;
    
    root.present()?;
    Ok(())
}

fn draw_bidder_performance_chart(
    canvas_id: &str,
    auction_history: &[AuctionResults],
) -> Result<(), Box<dyn std::error::Error>> {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .get_element_by_id(canvas_id)
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();
    
    let width = 800;
    let height = 400;
    canvas.set_width(width);
    canvas.set_height(height);
    
    let backend = CanvasBackend::new(canvas_id).expect("Cannot find canvas");
    let root = backend.into_drawing_area();
    root.fill(&WHITE)?;
    
    if auction_history.is_empty() {
        return Ok(());
    }
    
    // Calculate total revenue per bidder
    let mut bidder_revenues: HashMap<String, u32> = HashMap::new();
    
    for auction in auction_history {
        for result in &auction.bidder_results {
            *bidder_revenues.entry(result.name.clone()).or_insert(0) += result.revenue;
        }
    }
    
    let mut sorted_bidders: Vec<_> = bidder_revenues.into_iter().collect();
    sorted_bidders.sort_by(|a, b| b.1.cmp(&a.1));
    sorted_bidders.truncate(10); // Top 10 bidders
    
    if sorted_bidders.is_empty() {
        return Ok(());
    }
    
    let max_revenue = sorted_bidders[0].1 as i32;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Top Bidder Performance", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0f32..(sorted_bidders.len() as f32 + 1.0),
            0..(max_revenue + max_revenue / 10),
        )?;
    
    chart
        .configure_mesh()
        .x_desc("Bidder")
        .y_desc("Total Revenue ($)")
        .x_label_formatter(&|x| {
            let idx = *x as usize;
            if idx > 0 && idx <= sorted_bidders.len() {
                sorted_bidders[idx - 1].0.clone()
            } else {
                String::new()
            }
        })
        .draw()?;
    
    chart.draw_series(
        sorted_bidders
            .iter()
            .enumerate()
            .map(|(i, (_, revenue))| {
                Rectangle::new(
                    [((i as f32 + 0.5), 0), ((i as f32 + 1.5), *revenue as i32)],
                    GREEN.filled(),
                )
            }),
    )?;
    
    root.present()?;
    Ok(())
}

// Export functionality
fn download_file(filename: &str, content: &str, mime_type: &str) {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    
    // Create blob
    let blob_parts = js_sys::Array::new();
    blob_parts.push(&wasm_bindgen::JsValue::from_str(content));
    
    let mut blob_property_bag = web_sys::BlobPropertyBag::new();
    blob_property_bag.set_type(mime_type);  // Updated from type_ to set_type
    
    let blob = web_sys::Blob::new_with_str_sequence_and_options(&blob_parts, &blob_property_bag).unwrap();
    
    // Create download URL
    let url = web_sys::Url::create_object_url_with_blob(&blob).unwrap();
    
    // Create anchor element and trigger download
    let a = document.create_element("a").unwrap()
        .dyn_into::<web_sys::HtmlAnchorElement>().unwrap();
    a.set_href(&url);
    a.set_download(filename);
    a.click();
    
    // Clean up
    web_sys::Url::revoke_object_url(&url).unwrap();
}

fn export_auction_history_json(auction_history: &[AuctionResults]) {
    let json_data = serde_json::json!({
        "exportDate": chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        "totalAuctions": auction_history.len(),
        "auctions": auction_history.iter().map(|auction| {
            serde_json::json!({
                "auctionNumber": auction.auction_number,
                "timestamp": auction.timestamp,
                "trustAssets": auction.trust_assets,
                "redemptionPool": auction.redemption_pool,
                "redemptionPercentage": auction.redemption_percentage,
                "marketCondition": auction.market_condition.name(),
                "totalSpent": auction.total_spent,
                "remainingPool": auction.remaining_pool,
                "tokensPurchased": auction.tokens_purchased,
                "averagePrice": auction.average_purchase_price,
                "totalBids": auction.total_bids,
                "clearedBids": auction.cleared_bids.len(),
                "rejectedBids": auction.rejected_bids.len(),
                "bidderResults": auction.bidder_results.iter().map(|result| {
                    serde_json::json!({
                        "name": result.name,
                        "profile": result.profile.to_string(),
                        "initialTokens": result.initial_tokens,
                        "tokensSold": result.tokens_sold,
                        "tokensLeft": result.tokens_left,
                        "tokensHeld": result.tokens_held,
                        "revenue": result.revenue,
                        "successRate": result.success_rate
                    })
                }).collect::<Vec<_>>()
            })
        }).collect::<Vec<_>>()
    });
    
    let content = serde_json::to_string_pretty(&json_data).unwrap();
    let filename = format!("auction-history-{}.json", chrono::Local::now().format("%Y%m%d-%H%M%S"));
    download_file(&filename, &content, "application/json");
}

fn export_auction_history_csv(auction_history: &[AuctionResults]) {
    let mut csv_content = String::from("Auction,Timestamp,Trust Assets,Redemption Pool,Redemption %,Market,Total Spent,Tokens Purchased,Average Price,Total Bids,Cleared Bids,Rejected Bids,Pool Utilization %\n");
    
    for auction in auction_history {
        let pool_utilization = (auction.total_spent as f64 / auction.redemption_pool as f64 * 100.0) as u32;
        csv_content.push_str(&format!(
            "{},{},{},{},{},{},{},{},{:.2},{},{},{},{}\n",
            auction.auction_number,
            auction.timestamp,
            auction.trust_assets,
            auction.redemption_pool,
            auction.redemption_percentage,
            auction.market_condition.name(),
            auction.total_spent,
            auction.tokens_purchased,
            auction.average_purchase_price,
            auction.total_bids,
            auction.cleared_bids.len(),
            auction.rejected_bids.len(),
            pool_utilization
        ));
    }
    
    let filename = format!("auction-history-{}.csv", chrono::Local::now().format("%Y%m%d-%H%M%S"));
    download_file(&filename, &csv_content, "text/csv");
}

fn export_bidder_summary_csv(auction_history: &[AuctionResults]) {
    let mut csv_content = String::from("Bidder,Profile,Total Revenue,Total Tokens Sold,Auctions Participated,Average Success Rate\n");
    
    // Aggregate bidder data across all auctions
    let mut bidder_totals: HashMap<String, (String, u32, usize, Vec<f64>)> = HashMap::new();
    
    for auction in auction_history {
        for result in &auction.bidder_results {
            let entry = bidder_totals.entry(result.name.clone())
                .or_insert((result.profile.to_string(), 0, 0, Vec::new()));
            entry.1 += result.revenue;
            entry.2 += result.tokens_sold;
            entry.3.push(result.success_rate);
        }
    }
    
    // Sort by total revenue
    let mut sorted_bidders: Vec<_> = bidder_totals.into_iter().collect();
    sorted_bidders.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    
    for (name, (profile, revenue, tokens_sold, success_rates)) in sorted_bidders {
        let avg_success_rate = success_rates.iter().sum::<f64>() / success_rates.len() as f64;
        csv_content.push_str(&format!(
            "{},{},{},{},{},{:.1}\n",
            name,
            profile,
            revenue,
            tokens_sold,
            success_rates.len(),
            avg_success_rate
        ));
    }
    
    let filename = format!("bidder-summary-{}.csv", chrono::Local::now().format("%Y%m%d-%H%M%S"));
    download_file(&filename, &csv_content, "text/csv");
}

#[component]
pub fn App() -> impl IntoView {
    // UI state
    let (active_tab, set_active_tab) = create_signal(ActiveTab::Simulation);
    let (dark_mode, set_dark_mode) = create_signal(false);
    let (selected_auction_for_details, set_selected_auction_for_details) = create_signal(None::<AuctionResults>);
    
    // Core configuration
    let (trust_assets, set_trust_assets) = create_signal(350000u32);
    let (redemption_percentage, set_redemption_percentage) = create_signal(5u32);
    let (num_bidders, set_num_bidders) = create_signal(30usize);
    let (total_tokens, set_total_tokens) = create_signal(60000usize);
    let (use_profiles, set_use_profiles) = create_signal(true);
    let (market_condition, set_market_condition) = create_signal(MarketCondition::Neutral);
    
    // Growth rates
    let (trust_assets_growth, set_trust_assets_growth) = create_signal(30u32);
    let (redemption_percentage_growth, set_redemption_percentage_growth) = create_signal(20u32);
    let (bidder_growth, set_bidder_growth) = create_signal(30u32);
    let (token_growth, set_token_growth) = create_signal(40u32);
    
    // Auction state
    let (auction_number, set_auction_number) = create_signal(1usize);
    let (bidders, set_bidders) = create_signal(Vec::<Bidder>::new());
    let (auction_results, set_auction_results) = create_signal(None::<AuctionResults>);
    let (auction_history, set_auction_history) = create_signal(Vec::<AuctionResults>::new());
    let (charts_loading, set_charts_loading) = create_signal(false);
    
    // Computed redemption pool
    let redemption_pool = move || {
        (trust_assets.get() as f64 * (redemption_percentage.get() as f64 / 100.0)) as u32
    };

    // Generate initial bidders
    let generate_bidders_action = move || {
        set_auction_results.set(None);
        let new_bidders = generate_initial_bidders(
            num_bidders.get(),
            total_tokens.get(),
            use_profiles.get()
        );
        set_bidders.set(new_bidders);
    };

    // Run auction
    let run_auction_action = move || {
        let mut current_bidders = bidders.get();
        
        // Generate bids for current auction
        generate_bids_for_bidders(
            &mut current_bidders,
            use_profiles.get(),
            market_condition.get(),
            auction_number.get()
        );
        
        // Run auction with learning
        let results = run_auction_with_learning(
            &mut current_bidders, 
            redemption_pool(),
            auction_number.get(),
            trust_assets.get(),
            redemption_percentage.get(),
            market_condition.get()
        );
        
        set_auction_results.set(Some(results.clone()));
        set_bidders.set(current_bidders);
        
        // Add to history
        let mut history = auction_history.get();
        history.push(results.clone());
        set_auction_history.set(history);
        
        // Set as selected for bid details
        set_selected_auction_for_details.set(Some(results));
    };
    
    // Next auction with growth
    let next_auction_action = move || {
        // Apply growth rates
        let new_trust_assets = (trust_assets.get() as f64 * (1.0 + trust_assets_growth.get() as f64 / 100.0)) as u32;
        let new_redemption_percentage = std::cmp::min(
            100,
            (redemption_percentage.get() as f64 * (1.0 + redemption_percentage_growth.get() as f64 / 100.0)) as u32
        );
        let new_bidder_count = (num_bidders.get() as f64 * (1.0 + bidder_growth.get() as f64 / 100.0)) as usize;
        let new_total_tokens = (total_tokens.get() as f64 * (1.0 + token_growth.get() as f64 / 100.0)) as usize;
        
        // Calculate new tokens to distribute
        let current_total_tokens: usize = bidders.get().iter().map(|b| b.tokens).sum();
        let new_tokens = new_total_tokens.saturating_sub(current_total_tokens);
        
        // Distribute new tokens to existing and new bidders
        let mut updated_bidders = bidders.get();
        distribute_new_tokens(&mut updated_bidders, new_tokens, new_bidder_count);
        
        // Update state
        set_trust_assets.set(new_trust_assets);
        set_redemption_percentage.set(new_redemption_percentage);
        set_num_bidders.set(new_bidder_count);
        set_total_tokens.set(new_total_tokens);
        set_auction_number.update(|n| *n += 1);
        set_bidders.set(updated_bidders);
        set_auction_results.set(None);
    };
    
    // Apply scenario template
    let apply_scenario = move |scenario: &ScenarioTemplate| {
        set_trust_assets.set(scenario.trust_assets);
        set_redemption_percentage.set(scenario.redemption_percentage);
        set_num_bidders.set(scenario.num_bidders);
        set_total_tokens.set(scenario.total_tokens);
        set_trust_assets_growth.set(scenario.trust_assets_growth);
        set_redemption_percentage_growth.set(scenario.redemption_percentage_growth);
        set_bidder_growth.set(scenario.bidder_growth);
        set_token_growth.set(scenario.token_growth);
        
        // Reset and regenerate with new settings
        set_auction_number.set(1);
        set_auction_history.set(Vec::new());
        let new_bidders = generate_initial_bidders(
            scenario.num_bidders,
            scenario.total_tokens,
            use_profiles.get()
        );
        set_bidders.set(new_bidders);
        set_auction_results.set(None);
    };

    // Initialize bidders on component mount
    create_effect(move |_| {
        generate_bidders_action();
    });
    
    view! {
        <div class=move || format!("min-h-screen transition-colors {}", 
            if dark_mode.get() { "bg-gray-900 text-white" } else { "bg-gray-50 text-gray-900" }
        )>
            <div class="max-w-7xl mx-auto p-4">
                <div class="flex justify-between items-center mb-6">
                    <h1 class="text-3xl font-bold">"Token Auction Simulator"</h1>
                    <div class="flex items-center gap-4">
                        <div class="flex items-center gap-2">
                            <span class="text-sm">"Market:"</span>
                            <select
                                value=move || format!("{:?}", market_condition.get())
                                on:change=move |ev| {
                                    let value = event_target_value(&ev);
                                    let condition = match value.as_str() {
                                        "Bull" => MarketCondition::Bull,
                                        "Bear" => MarketCondition::Bear,
                                        "Volatile" => MarketCondition::Volatile,
                                        _ => MarketCondition::Neutral,
                                    };
                                    set_market_condition.set(condition);
                                }
                                class=move || format!("px-3 py-1 rounded border {}",
                                    if dark_mode.get() { "bg-gray-800 border-gray-600" } else { "bg-white border-gray-300" }
                                )
                            >
                                <option value="Bull">"Bull Market"</option>
                                <option value="Neutral">"Neutral Market"</option>
                                <option value="Bear">"Bear Market"</option>
                                <option value="Volatile">"Volatile Market"</option>
                            </select>
                        </div>
                        <button
                            on:click=move |_| set_dark_mode.update(|d| *d = !*d)
                            class=move || format!("px-3 py-1 rounded {}",
                                if dark_mode.get() { "bg-gray-700 hover:bg-gray-600" } else { "bg-gray-200 hover:bg-gray-300" }
                            )
                        >
                            {move || if dark_mode.get() { "☀️" } else { "🌙" }}
                        </button>
                    </div>
                </div>
                
                // Tab Navigation
                <div class="mb-6">
                    <div class=move || format!("border-b {}",
                        if dark_mode.get() { "border-gray-700" } else { "border-gray-200" }
                    )>
                        <nav class="-mb-px flex space-x-8">
                            <button
                                on:click=move |_| set_active_tab.set(ActiveTab::Simulation)
                                class=move || {
                                    if active_tab.get() == ActiveTab::Simulation {
                                        "py-2 px-1 border-b-2 font-medium text-sm border-blue-500 text-blue-600"
                                    } else if dark_mode.get() {
                                        "py-2 px-1 border-b-2 font-medium text-sm border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-300"
                                    } else {
                                        "py-2 px-1 border-b-2 font-medium text-sm border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                                    }
                                }
                            >
                                "Simulation"
                            </button>
                            <button
                                on:click=move |_| set_active_tab.set(ActiveTab::Analytics)
                                class=move || {
                                    if active_tab.get() == ActiveTab::Analytics {
                                        "py-2 px-1 border-b-2 font-medium text-sm border-blue-500 text-blue-600"
                                    } else if dark_mode.get() {
                                        "py-2 px-1 border-b-2 font-medium text-sm border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-300"
                                    } else {
                                        "py-2 px-1 border-b-2 font-medium text-sm border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                                    }
                                }
                            >
                                "Analytics"
                            </button>
                            <button
                                on:click=move |_| set_active_tab.set(ActiveTab::BidDetails)
                                class=move || {
                                    if active_tab.get() == ActiveTab::BidDetails {
                                        "py-2 px-1 border-b-2 font-medium text-sm border-blue-500 text-blue-600"
                                    } else if dark_mode.get() {
                                        "py-2 px-1 border-b-2 font-medium text-sm border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-300"
                                    } else {
                                        "py-2 px-1 border-b-2 font-medium text-sm border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                                    }
                                }
                            >
                                "Bid Details"
                            </button>
                            <button
                                on:click=move |_| set_active_tab.set(ActiveTab::Configuration)
                                class=move || {
                                    if active_tab.get() == ActiveTab::Configuration {
                                        "py-2 px-1 border-b-2 font-medium text-sm border-blue-500 text-blue-600"
                                    } else if dark_mode.get() {
                                        "py-2 px-1 border-b-2 font-medium text-sm border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-300"
                                    } else {
                                        "py-2 px-1 border-b-2 font-medium text-sm border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                                    }
                                }
                            >
                                "Configuration"
                            </button>
                        </nav>
                    </div>
                </div>
                
                // Tab Content
                <div>
                    // Simulation Tab
                    <Show when=move || active_tab.get() == ActiveTab::Simulation>
                        <div>
                            // Status Bar
                            <div class=move || format!("p-4 rounded-lg mb-6 {}",
                                if dark_mode.get() { "bg-gray-800" } else { "bg-blue-50" }
                            )>
                                <div class="flex justify-between items-center mb-2">
                                    <h2 class="text-xl font-semibold">{format!("Auction #{}", auction_number.get())}</h2>
                                    <div class=move || format!("px-3 py-1 rounded text-sm {}",
                                        if dark_mode.get() { "bg-gray-700" } else { "bg-white" }
                                    )>
                                        {move || market_condition.get().name().to_string()}
                                    </div>
                                </div>
                                <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    <div class=move || format!("p-3 rounded shadow {}",
                                        if dark_mode.get() { "bg-gray-700" } else { "bg-white" }
                                    )>
                                        <div class=move || format!("text-sm {}",
                                            if dark_mode.get() { "text-gray-400" } else { "text-gray-600" }
                                        )>"Trust Assets"</div>
                                        <div class="text-lg font-semibold">{format!("${}", trust_assets.get())}</div>
                                    </div>
                                    <div class=move || format!("p-3 rounded shadow {}",
                                        if dark_mode.get() { "bg-gray-700" } else { "bg-white" }
                                    )>
                                        <div class=move || format!("text-sm {}",
                                            if dark_mode.get() { "text-gray-400" } else { "text-gray-600" }
                                        )>"Redemption Pool"</div>
                                        <div class="text-lg font-semibold">{format!("${} ({}%)", redemption_pool(), redemption_percentage.get())}</div>
                                    </div>
                                    <div class=move || format!("p-3 rounded shadow {}",
                                        if dark_mode.get() { "bg-gray-700" } else { "bg-white" }
                                    )>
                                        <div class=move || format!("text-sm {}",
                                            if dark_mode.get() { "text-gray-400" } else { "text-gray-600" }
                                        )>"Bidders"</div>
                                        <div class="text-lg font-semibold">{num_bidders.get()}</div>
                                    </div>
                                    <div class=move || format!("p-3 rounded shadow {}",
                                        if dark_mode.get() { "bg-gray-700" } else { "bg-white" }
                                    )>
                                        <div class=move || format!("text-sm {}",
                                            if dark_mode.get() { "text-gray-400" } else { "text-gray-600" }
                                        )>"Total Tokens"</div>
                                        <div class="text-lg font-semibold">{move || bidders.get().iter().map(|b| b.tokens).sum::<usize>()}</div>
                                    </div>
                                </div>
                            </div>
                            
                            // Action buttons
                            <div class=move || format!("p-4 rounded-lg shadow mb-6 {}",
                                if dark_mode.get() { "bg-gray-800" } else { "bg-white" }
                            )>
                                <h3 class="text-lg font-semibold mb-3">"Actions"</h3>
                                <div class="flex gap-3 flex-wrap">
                                    <button
                                        on:click=move |_| generate_bidders_action()
                                        class="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded transition"
                                    >
                                        "Regenerate Bidders"
                                    </button>
                                    
                                    <button
                                        on:click=move |_| run_auction_action()
                                        disabled=move || auction_results.get().is_some()
                                        class=move || format!("px-4 py-2 rounded transition {}",
                                            if auction_results.get().is_some() {
                                                if dark_mode.get() {
                                                    "bg-gray-600 cursor-not-allowed text-gray-400"
                                                } else {
                                                    "bg-gray-300 cursor-not-allowed text-gray-500"
                                                }
                                            } else {
                                                "bg-green-500 hover:bg-green-600 text-white"
                                            }
                                        )
                                    >
                                        {move || if auction_results.get().is_some() {
                                            "Auction Run - Advance to Next"
                                        } else {
                                            "Run Auction"
                                        }}
                                    </button>
                                    
                                    <Show when=move || auction_results.get().is_some()>
                                        <button
                                            on:click=move |_| next_auction_action()
                                            class="px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded transition"
                                        >
                                            "Next Auction"
                                        </button>
                                    </Show>
                                    
                                    <button
                                        on:click=move |_| {
                                            // Reset everything
                                            set_trust_assets.set(350000);
                                            set_redemption_percentage.set(5);
                                            set_num_bidders.set(30);
                                            set_total_tokens.set(60000);
                                            set_trust_assets_growth.set(30);
                                            set_redemption_percentage_growth.set(20);
                                            set_bidder_growth.set(30);
                                            set_token_growth.set(40);
                                            set_auction_number.set(1);
                                            set_auction_history.set(Vec::new());
                                            set_market_condition.set(MarketCondition::Neutral);
                                            // Generate fresh bidders
                                            let new_bidders = generate_initial_bidders(30, 60000, true);
                                            set_bidders.set(new_bidders);
                                        }
                                        class="px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded transition"
                                    >
                                        "Reset"
                                    </button>
                                </div>
                            </div>
                            
                            // Current Bidders
                            <div class=move || format!("p-4 rounded-lg shadow mb-6 {}",
                                if dark_mode.get() { "bg-gray-800" } else { "bg-white" }
                            )>
                                <h3 class="text-lg font-semibold mb-3">"Current Bidders"</h3>
                                <div class="overflow-x-auto">
                                    <table class="min-w-full">
                                        <thead>
                                            <tr class=move || if dark_mode.get() { "bg-gray-700" } else { "bg-gray-100" }>
                                                <th class="px-4 py-2 text-left text-sm font-medium">"Bidder"</th>
                                                <th class="px-4 py-2 text-left text-sm font-medium">"Profile"</th>
                                                <th class="px-4 py-2 text-left text-sm font-medium">"Tokens"</th>
                                                <th class="px-4 py-2 text-left text-sm font-medium">"Success Rate"</th>
                                                <th class="px-4 py-2 text-left text-sm font-medium">"Total Revenue"</th>
                                                <th class="px-4 py-2 text-left text-sm font-medium">"Auctions"</th>
                                                <th class="px-4 py-2 text-left text-sm font-medium">"Learning"</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {move || bidders.get().into_iter().map(|bidder| {
                                                view! {
                                                    <tr class=move || format!("border-b {}",
                                                        if dark_mode.get() { "border-gray-700" } else { "border-gray-200" }
                                                    )>
                                                        <td class="px-4 py-2">{bidder.name}</td>
                                                        <td class="px-4 py-2">
                                                            <span class=format!("px-2 py-1 {} text-white text-xs rounded", bidder.profile.color_class())>
                                                                {bidder.profile.to_string()}
                                                            </span>
                                                        </td>
                                                        <td class="px-4 py-2">{bidder.tokens}</td>
                                                        <td class="px-4 py-2">
                                                            <span class=format!("px-2 py-1 rounded text-xs {}", 
                                                                if bidder.success_rate > 70.0 { "bg-green-100 text-green-800" }
                                                                else if bidder.success_rate > 30.0 { "bg-yellow-100 text-yellow-800" }
                                                                else { "bg-red-100 text-red-800" }
                                                            )>
                                                                {format!("{:.0}%", bidder.success_rate)}
                                                            </span>
                                                        </td>
                                                        <td class="px-4 py-2">{format!("${}", bidder.total_revenue)}</td>
                                                        <td class="px-4 py-2">{bidder.auction_history.len()}</td>
                                                        <td class="px-4 py-2">
                                                            <div class="text-xs text-gray-500">
                                                                {format!("Price: ${}-${}", 
                                                                    bidder.adapted_price_range.0,
                                                                    bidder.adapted_price_range.1
                                                                )}
                                                            </div>
                                                        </td>
                                                    </tr>
                                                }
                                            }).collect::<Vec<_>>()}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                            
                            // Auction Results
                            <Show when=move || auction_results.get().is_some()>
                                {move || {
                                    let results = auction_results.get().unwrap();
                                    view! {
                                        <div class=move || format!("p-6 rounded-lg shadow mb-6 {}",
                                            if dark_mode.get() { "bg-gray-800" } else { "bg-white" }
                                        )>
                                            <h2 class="text-xl font-semibold mb-4">"Auction Results"</h2>
                                            
                                            // Summary
                                            <div class="mb-6">
                                                <h3 class="text-lg font-semibold mb-3">"Summary"</h3>
                                                <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
                                                    <div class=move || format!("p-3 rounded {}",
                                                        if dark_mode.get() { "bg-gray-700" } else { "bg-gray-50" }
                                                    )>
                                                        <div class=move || format!("text-sm {}",
                                                            if dark_mode.get() { "text-gray-400" } else { "text-gray-600" }
                                                        )>"Trust Pool"</div>
                                                        <div class="text-lg font-semibold">{format!("${}", redemption_pool())}</div>
                                                    </div>
                                                    <div class=move || format!("p-3 rounded {}",
                                                        if dark_mode.get() { "bg-gray-700" } else { "bg-gray-50" }
                                                    )>
                                                        <div class=move || format!("text-sm {}",
                                                            if dark_mode.get() { "text-gray-400" } else { "text-gray-600" }
                                                        )>"Money Spent"</div>
                                                        <div class="text-lg font-semibold">{format!("${}", results.total_spent)}</div>
                                                    </div>
                                                    <div class=move || format!("p-3 rounded {}",
                                                        if dark_mode.get() { "bg-gray-700" } else { "bg-gray-50" }
                                                    )>
                                                        <div class=move || format!("text-sm {}",
                                                            if dark_mode.get() { "text-gray-400" } else { "text-gray-600" }
                                                        )>"Tokens Purchased"</div>
                                                        <div class="text-lg font-semibold">{results.tokens_purchased}</div>
                                                    </div>
                                                    <div class=move || format!("p-3 rounded {}",
                                                        if dark_mode.get() { "bg-gray-700" } else { "bg-gray-50" }
                                                    )>
                                                        <div class=move || format!("text-sm {}",
                                                            if dark_mode.get() { "text-gray-400" } else { "text-gray-600" }
                                                        )>"Avg. Price"</div>
                                                        <div class="text-lg font-semibold">{format!("${:.0}", results.average_purchase_price)}</div>
                                                    </div>
                                                </div>
                                            </div>
                                            
                                            // Bidder Results Table
                                            <div class="mb-6">
                                                <h3 class="text-lg font-semibold mb-3">"Bidder Results"</h3>
                                                <div class="overflow-x-auto">
                                                    <table class="min-w-full">
                                                        <thead>
                                                            <tr class=move || if dark_mode.get() { "bg-gray-700" } else { "bg-gray-100" }>
                                                                <th class="px-4 py-2 text-left text-sm font-medium">"Bidder"</th>
                                                                <th class="px-4 py-2 text-left text-sm font-medium">"Profile"</th>
                                                                <th class="px-4 py-2 text-left text-sm font-medium">"Initial"</th>
                                                                <th class="px-4 py-2 text-left text-sm font-medium">"Offered"</th>
                                                                <th class="px-4 py-2 text-left text-sm font-medium">"Sold"</th>
                                                                <th class="px-4 py-2 text-left text-sm font-medium">"Held"</th>
                                                                <th class="px-4 py-2 text-left text-sm font-medium">"Left"</th>
                                                                <th class="px-4 py-2 text-left text-sm font-medium">"New"</th>
                                                                <th class="px-4 py-2 text-left text-sm font-medium">"Revenue"</th>
                                                                <th class="px-4 py-2 text-left text-sm font-medium">"Success"</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            {results.bidder_results.clone().into_iter().map(|result| {
                                                                view! {
                                                                    <tr class=move || format!("border-b {}",
                                                                        if dark_mode.get() { "border-gray-700" } else { "border-gray-200" }
                                                                    )>
                                                                        <td class="px-4 py-2">{result.name}</td>
                                                                        <td class="px-4 py-2">
                                                                            <span class=format!("px-2 py-1 {} text-white text-xs rounded", result.profile.color_class())>
                                                                                {result.profile.to_string()}
                                                                            </span>
                                                                        </td>
                                                                        <td class="px-4 py-2">{result.initial_tokens}</td>
                                                                        <td class="px-4 py-2">{result.initial_tokens - result.tokens_held}</td>
                                                                        <td class="px-4 py-2">{result.tokens_sold}</td>
                                                                        <td class="px-4 py-2">{result.tokens_held}</td>
                                                                        <td class="px-4 py-2">{result.tokens_left}</td>
                                                                        <td class="px-4 py-2">
                                                                            {if result.new_tokens > 0 {
                                                                                format!("+{}", result.new_tokens)
                                                                            } else {
                                                                                "".to_string()
                                                                            }}
                                                                        </td>
                                                                        <td class="px-4 py-2">{format!("${}", result.revenue)}</td>
                                                                        <td class="px-4 py-2">
                                                                            <span class=format!("px-2 py-1 rounded text-xs {}", 
                                                                                if result.success_rate > 70.0 { "bg-green-100 text-green-800" }
                                                                                else if result.success_rate > 30.0 { "bg-yellow-100 text-yellow-800" }
                                                                                else { "bg-red-100 text-red-800" }
                                                                            )>
                                                                                {format!("{:.0}%", result.success_rate)}
                                                                            </span>
                                                                        </td>
                                                                    </tr>
                                                                }
                                                            }).collect::<Vec<_>>()}
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </div>
                                        </div>
                                    }
                                }}
                            </Show>
                            
                            // Auction History
                            <Show when=move || !auction_history.get().is_empty()>
                                <div class=move || format!("p-6 rounded-lg shadow {}",
                                    if dark_mode.get() { "bg-gray-800" } else { "bg-white" }
                                )>
                                    <h2 class="text-xl font-semibold mb-4">"Auction History"</h2>
                                    <div class="overflow-x-auto">
                                        <table class="min-w-full">
                                            <thead>
                                                <tr class=move || if dark_mode.get() { "bg-gray-700" } else { "bg-gray-100" }>
                                                    <th class="px-4 py-2 text-left text-sm font-medium">"Auction"</th>
                                                    <th class="px-4 py-2 text-left text-sm font-medium">"Market"</th>
                                                    <th class="px-4 py-2 text-left text-sm font-medium">"Trust Assets"</th>
                                                    <th class="px-4 py-2 text-left text-sm font-medium">"Pool"</th>
                                                    <th class="px-4 py-2 text-left text-sm font-medium">"Spent"</th>
                                                    <th class="px-4 py-2 text-left text-sm font-medium">"Tokens Sold"</th>
                                                    <th class="px-4 py-2 text-left text-sm font-medium">"Avg Price"</th>
                                                    <th class="px-4 py-2 text-left text-sm font-medium">"Timestamp"</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {move || auction_history.get().into_iter().map(|auction| {
                                                    let auction_clone = auction.clone();
                                                    let market_condition_name = auction.market_condition.name().to_string();
                                                    let timestamp = auction.timestamp.clone();
                                                    let is_selected = selected_auction_for_details.get()
                                                        .map(|selected| selected.auction_number == auction.auction_number)
                                                        .unwrap_or(false);
                                                    
                                                    view! {
                                                        <tr class=move || format!("border-b cursor-pointer hover:bg-gray-50 {}",
                                                            if dark_mode.get() { "border-gray-700 hover:bg-gray-700" } else { "border-gray-200" }
                                                        )
                                                            on:click=move |_| {
                                                                set_selected_auction_for_details.set(Some(auction_clone.clone()));
                                                                set_active_tab.set(ActiveTab::BidDetails);
                                                            }
                                                        >
                                                            <td class="px-4 py-2">{format!("#{}", auction.auction_number)}</td>
                                                            <td class="px-4 py-2">
                                                                <span class="px-2 py-1 bg-gray-100 text-xs rounded">
                                                                    {market_condition_name}
                                                                </span>
                                                            </td>
                                                            <td class="px-4 py-2">{format!("${}", auction.trust_assets)}</td>
                                                            <td class="px-4 py-2">{format!("${}", auction.redemption_pool)}</td>
                                                            <td class="px-4 py-2">{format!("${}", auction.total_spent)}</td>
                                                            <td class="px-4 py-2">{auction.tokens_purchased}</td>
                                                            <td class="px-4 py-2">{format!("${:.0}", auction.average_purchase_price)}</td>
                                                            <td class="px-4 py-2 text-xs">{timestamp.clone()}</td>
                                                        </tr>
                                                    }
                                                }).collect::<Vec<_>>()}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </Show>
                        </div>
                    </Show>
                    
                    // Analytics Tab
                    <Show when=move || active_tab.get() == ActiveTab::Analytics>
                        <div class=move || format!("p-6 rounded-lg shadow {}",
                            if dark_mode.get() { "bg-gray-800" } else { "bg-white" }
                        )>
                            <div class="flex justify-between items-center mb-6">
                                <h2 class="text-2xl font-semibold">"Market Analytics"</h2>
                                {move || if !auction_history.get().is_empty() {
                                    view! {
                                        <div class="flex gap-2">
                                            <button
                                                on:click=move |_| {
                                                    let history = auction_history.get();
                                                    export_auction_history_json(&history);
                                                }
                                                class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded text-sm transition"
                                            >
                                                "Export JSON"
                                            </button>
                                            <button
                                                on:click=move |_| {
                                                    let history = auction_history.get();
                                                    export_auction_history_csv(&history);
                                                }
                                                class="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded text-sm transition"
                                            >
                                                "Export History CSV"
                                            </button>
                                            <button
                                                on:click=move |_| {
                                                    let history = auction_history.get();
                                                    export_bidder_summary_csv(&history);
                                                }
                                                class="bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded text-sm transition"
                                            >
                                                "Export Bidders CSV"
                                            </button>
                                        </div>
                                    }.into_view()
                                } else {
                                    view! { <div></div> }.into_view()
                                }}
                            </div>
                            
                            {move || if auction_history.get().is_empty() {
                                view! {
                                    <div class="text-center py-12">
                                        <div class="text-gray-400 text-6xl mb-4">"📈"</div>
                                        <h3 class=move || format!("text-xl font-medium mb-2 {}",
                                            if dark_mode.get() { "text-gray-300" } else { "text-gray-600" }
                                        )>"No Data Available"</h3>
                                        <p class=move || format!("mb-6 {}",
                                            if dark_mode.get() { "text-gray-400" } else { "text-gray-500" }
                                        )>"Run some auctions to see detailed analytics"</p>
                                        <button
                                            on:click=move |_| set_active_tab.set(ActiveTab::Simulation)
                                            class="bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-lg font-medium"
                                        >
                                            "Go to Simulation"
                                        </button>
                                    </div>
                                }.into_view()
                            } else {
                                // Draw charts when history is available
                                create_effect(move |_| {
                                    let history = auction_history.get();
                                    if !history.is_empty() && active_tab.get() == ActiveTab::Analytics {
                                        set_charts_loading.set(true);
                                        // Give DOM time to render canvases
                                        let history_clone = history.clone();
                                        set_timeout(
                                            move || {
                                                let _ = draw_price_history_chart("price-history-chart", &history_clone);
                                                let _ = draw_trading_volume_chart("trading-volume-chart", &history_clone);
                                                let _ = draw_bidder_performance_chart("bidder-performance-chart", &history_clone);
                                                set_charts_loading.set(false);
                                            },
                                            std::time::Duration::from_millis(100),
                                        );
                                    }
                                });
                                
                                view! {
                                    <div class="space-y-8">
                                        // Summary Statistics
                                        <div>
                                            <h3 class="text-lg font-semibold mb-4">"Summary Statistics"</h3>
                                            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                                                {move || {
                                                    let history = auction_history.get();
                                                    let total_tokens: usize = history.iter().map(|a| a.tokens_purchased).sum();
                                                    let total_spent: u32 = history.iter().map(|a| a.total_spent).sum();
                                                    let avg_price = if total_tokens > 0 {
                                                        total_spent / total_tokens as u32
                                                    } else {
                                                        0
                                                    };
                                                    
                                                    view! {
                                                        <>
                                                            <div class=move || format!("p-4 rounded {}",
                                                                if dark_mode.get() { "bg-gray-700" } else { "bg-gray-50" }
                                                            )>
                                                                <div class="text-2xl font-bold text-blue-600">{history.len()}</div>
                                                                <div class=move || format!("text-sm {}",
                                                                    if dark_mode.get() { "text-gray-400" } else { "text-gray-600" }
                                                                )>"Total Auctions"</div>
                                                            </div>
                                                            <div class=move || format!("p-4 rounded {}",
                                                                if dark_mode.get() { "bg-gray-700" } else { "bg-gray-50" }
                                                            )>
                                                                <div class="text-2xl font-bold text-green-600">{total_tokens}</div>
                                                                <div class=move || format!("text-sm {}",
                                                                    if dark_mode.get() { "text-gray-400" } else { "text-gray-600" }
                                                                )>"Total Tokens Traded"</div>
                                                            </div>
                                                            <div class=move || format!("p-4 rounded {}",
                                                                if dark_mode.get() { "bg-gray-700" } else { "bg-gray-50" }
                                                            )>
                                                                <div class="text-2xl font-bold text-purple-600">{format!("${}", avg_price)}</div>
                                                                <div class=move || format!("text-sm {}",
                                                                    if dark_mode.get() { "text-gray-400" } else { "text-gray-600" }
                                                                )>"Overall Avg Price"</div>
                                                            </div>
                                                        </>
                                                    }
                                                }}
                                            </div>
                                        </div>

                                        // Price History Chart
                                        <div>
                                            <h3 class="text-lg font-semibold mb-4">"Price & Pool Trends"</h3>
                                            <div class=move || format!("p-4 rounded relative {}",
                                                if dark_mode.get() { "bg-gray-700" } else { "bg-gray-100" }
                                            )>
                                                <Show when=move || charts_loading.get()>
                                                    <div class="absolute inset-0 flex items-center justify-center bg-black bg-opacity-10 rounded">
                                                        <div class="text-lg">"Loading charts..."</div>
                                                    </div>
                                                </Show>
                                                <canvas id="price-history-chart"></canvas>
                                            </div>
                                        </div>
                                        
                                        // Trading Volume Chart
                                        <div>
                                            <h3 class="text-lg font-semibold mb-4">"Trading Volume"</h3>
                                            <div class=move || format!("p-4 rounded relative {}",
                                                if dark_mode.get() { "bg-gray-700" } else { "bg-gray-100" }
                                            )>
                                                <Show when=move || charts_loading.get()>
                                                    <div class="absolute inset-0 flex items-center justify-center bg-black bg-opacity-10 rounded">
                                                        <div class="text-lg">"Loading charts..."</div>
                                                    </div>
                                                </Show>
                                                <canvas id="trading-volume-chart"></canvas>
                                            </div>
                                        </div>
                                        
                                        // Bidder Performance Chart
                                        <div>
                                            <h3 class="text-lg font-semibold mb-4">"Top Bidder Performance"</h3>
                                            <div class=move || format!("p-4 rounded relative {}",
                                                if dark_mode.get() { "bg-gray-700" } else { "bg-gray-100" }
                                            )>
                                                <Show when=move || charts_loading.get()>
                                                    <div class="absolute inset-0 flex items-center justify-center bg-black bg-opacity-10 rounded">
                                                        <div class="text-lg">"Loading charts..."</div>
                                                    </div>
                                                </Show>
                                                <canvas id="bidder-performance-chart"></canvas>
                                            </div>
                                        </div>
                                    </div>
                                }.into_view()
                            }}
                        </div>
                    </Show>
                    
                    // Bid Details Tab
                    <Show when=move || active_tab.get() == ActiveTab::BidDetails>
                        <div class=move || format!("p-6 rounded-lg shadow {}",
                            if dark_mode.get() { "bg-gray-800" } else { "bg-white" }
                        )>
                            <h2 class="text-2xl font-semibold mb-6">"Detailed Bid Analysis"</h2>
                            
                            {move || if auction_history.get().is_empty() {
                                view! {
                                    <div class="text-center py-12">
                                        <div class="text-gray-400 text-6xl mb-4">"📊"</div>
                                        <h3 class=move || format!("text-xl font-medium mb-2 {}",
                                            if dark_mode.get() { "text-gray-300" } else { "text-gray-600" }
                                        )>"No Auction Data Available"</h3>
                                        <p class=move || format!("mb-6 {}",
                                            if dark_mode.get() { "text-gray-400" } else { "text-gray-500" }
                                        )>"Run an auction to see detailed bid information"</p>
                                        <button
                                            on:click=move |_| set_active_tab.set(ActiveTab::Simulation)
                                            class="bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-lg font-medium"
                                        >
                                            "Go to Simulation"
                                        </button>
                                    </div>
                                }.into_view()
                            } else {
                                view! {
                                    <div>
                                        // Auction Selector
                                        <div class="mb-6">
                                            <h3 class="text-lg font-semibold mb-3">"Select Auction to Analyze"</h3>
                                            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                                                {move || auction_history.get().into_iter().map(|auction| {
                                                    let auction_clone = auction.clone();
                                                    let market_condition_name = auction.market_condition.name().to_string();
                                                    let timestamp = auction.timestamp.clone();
                                                    let is_selected = selected_auction_for_details.get()
                                                        .map(|selected| selected.auction_number == auction.auction_number)
                                                        .unwrap_or(false);
                                                    
                                                    view! {
                                                        <button
                                                            on:click=move |_| {
                                                                set_selected_auction_for_details.set(Some(auction_clone.clone()));
                                                            }
                                                            class=move || format!(
                                                                "p-4 rounded-lg border-2 text-left transition-all {}",
                                                                if is_selected {
                                                                    if dark_mode.get() {
                                                                        "border-blue-500 bg-blue-900 shadow-lg transform scale-105"
                                                                    } else {
                                                                        "border-blue-500 bg-blue-50 shadow-lg transform scale-105"
                                                                    }
                                                                } else if dark_mode.get() {
                                                                    "border-gray-600 bg-gray-700 hover:border-blue-400 hover:bg-gray-600"
                                                                } else {
                                                                    "border-gray-200 bg-white hover:border-blue-300 hover:bg-gray-50"
                                                                }
                                                            )
                                                        >
                                                            <div class="flex justify-between items-start mb-2">
                                                                <h4 class="font-semibold">{format!("Auction #{}", auction.auction_number)}</h4>
                                                                <span class=move || format!("text-xs px-2 py-1 rounded {}",
                                                                    match auction.market_condition {
                                                                        MarketCondition::Bull => "bg-green-100 text-green-800",
                                                                        MarketCondition::Bear => "bg-red-100 text-red-800",
                                                                        MarketCondition::Volatile => "bg-yellow-100 text-yellow-800",
                                                                        MarketCondition::Neutral => "bg-gray-100 text-gray-800",
                                                                    }
                                                                )>
                                                                    {market_condition_name.clone()}
                                                                </span>
                                                            </div>
                                                            <div class=move || format!("text-sm space-y-1 {}",
                                                                if dark_mode.get() { "text-gray-300" } else { "text-gray-600" }
                                                            )>
                                                                <div class="flex justify-between">
                                                                    <span>"Tokens:"</span>
                                                                    <span class="font-medium">{auction.tokens_purchased}</span>
                                                                </div>
                                                                <div class="flex justify-between">
                                                                    <span>"Avg Price:"</span>
                                                                    <span class="font-medium">{format!("${:.0}", auction.average_purchase_price)}</span>
                                                                </div>
                                                                <div class="flex justify-between">
                                                                    <span>"Pool Used:"</span>
                                                                    <span class="font-medium">
                                                                        {format!("{}%", (auction.total_spent as f64 / auction.redemption_pool as f64 * 100.0) as u32)}
                                                                    </span>
                                                                </div>
                                                            </div>
                                                            <div class=move || format!("text-xs mt-2 {}",
                                                                if dark_mode.get() { "text-gray-400" } else { "text-gray-500" }
                                                            )>
                                                                {timestamp.clone()}
                                                            </div>
                                                        </button>
                                                    }
                                                }).collect::<Vec<_>>()}
                                            </div>
                                        </div>
                                        
                                        // Selected Auction Details
                                        {move || if let Some(auction) = selected_auction_for_details.get() {
                                            let market_condition_name = auction.market_condition.name().to_string();
                                            view! {
                                                <div>
                                                    // Auction Summary
                                                    <div class=move || format!("p-4 rounded-lg mb-6 {}",
                                                        if dark_mode.get() { "bg-gray-800" } else { "bg-white" }
                                                    )>
                                                        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                                                            <div>"Trust Assets: "{format!("${}", auction.trust_assets)}</div>
                                                            <div>"Redemption Pool: "{format!("${}", auction.redemption_pool)}</div>
                                                            <div>"Market: "{market_condition_name}</div>
                                                            <div>"Time: "{auction.timestamp.clone()}</div>
                                                        </div>
                                                    </div>
                                                    
                                                    // Bid Lists
                                                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                                        // Accepted Bids
                                                        <div class=move || format!("p-4 rounded-lg border {}",
                                                            if dark_mode.get() { "bg-gray-700 border-green-600" } else { "bg-green-50 border-green-200" }
                                                        )>
                                                            <h4 class=move || format!("text-lg font-semibold mb-3 {}",
                                                                if dark_mode.get() { "text-green-400" } else { "text-green-800" }
                                                            )>
                                                                {format!("Accepted Bids ({})", auction.cleared_bids.len())}
                                                            </h4>
                                                            <div class="space-y-2 max-h-96 overflow-y-auto">
                                                                {auction.cleared_bids.clone().into_iter().enumerate().map(|(idx, bid)| {
                                                                    view! {
                                                                        <div class=move || format!("p-3 rounded border {}",
                                                                            if dark_mode.get() { "bg-gray-800 border-green-700" } else { "bg-white border-green-100" }
                                                                        )>
                                                                            <div class="flex justify-between items-center">
                                                                                <div>
                                                                                    <div class=move || format!("font-medium {}",
                                                                                        if dark_mode.get() { "text-green-400" } else { "text-green-800" }
                                                                                    )>{bid.bidder_name}</div>
                                                                                    <div class=move || format!("text-sm {}",
                                                                                        if dark_mode.get() { "text-gray-400" } else { "text-gray-600" }
                                                                                    )>{format!("Rank: #{}", idx + 1)}</div>
                                                                                </div>
                                                                                <div class="text-right">
                                                                                    <div class="text-lg font-bold text-green-600">{format!("${}", bid.price)}</div>
                                                                                    <div class="text-xs text-gray-500">"Accepted"</div>
                                                                                </div>
                                                                            </div>
                                                                        </div>
                                                                    }
                                                                }).collect::<Vec<_>>()}
                                                            </div>
                                                        </div>
                                                        
                                                        // Rejected Bids
                                                        <div class=move || format!("p-4 rounded-lg border {}",
                                                            if dark_mode.get() { "bg-gray-700 border-red-600" } else { "bg-red-50 border-red-200" }
                                                        )>
                                                            <h4 class=move || format!("text-lg font-semibold mb-3 {}",
                                                                if dark_mode.get() { "text-red-400" } else { "text-red-800" }
                                                            )>
                                                                {format!("Rejected Bids ({})", auction.rejected_bids.len())}
                                                            </h4>
                                                            <div class="space-y-2 max-h-96 overflow-y-auto">
                                                                {auction.rejected_bids.clone().into_iter().enumerate().map(|(idx, bid)| {
                                                                    view! {
                                                                        <div class=move || format!("p-3 rounded border {}",
                                                                            if dark_mode.get() { "bg-gray-800 border-red-700" } else { "bg-white border-red-100" }
                                                                        )>
                                                                            <div class="flex justify-between items-center">
                                                                                <div>
                                                                                    <div class=move || format!("font-medium {}",
                                                                                        if dark_mode.get() { "text-red-400" } else { "text-red-800" }
                                                                                    )>{bid.bidder_name}</div>
                                                                                    <div class=move || format!("text-sm {}",
                                                                                        if dark_mode.get() { "text-gray-400" } else { "text-gray-600" }
                                                                                    )>
                                                                                        {format!("Would have been rank #{}", auction.cleared_bids.len() + idx + 1)}
                                                                                    </div>
                                                                                </div>
                                                                                <div class="text-right">
                                                                                    <div class="text-lg font-bold text-red-600">{format!("${}", bid.price)}</div>
                                                                                    <div class="text-xs text-gray-500">"Rejected"</div>
                                                                                </div>
                                                                            </div>
                                                                        </div>
                                                                    }
                                                                }).collect::<Vec<_>>()}
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            }.into_view()
                                        } else {
                                            view! {
                                                <div class=move || format!("mt-8 p-8 text-center rounded-lg border-2 border-dashed {}",
                                                    if dark_mode.get() { "border-gray-600 text-gray-400" } else { "border-gray-300 text-gray-500" }
                                                )>
                                                    <div class="text-5xl mb-4">"👆"</div>
                                                    <p class="text-lg">"Select an auction above to view detailed bid information"</p>
                                                </div>
                                            }.into_view()
                                        }}
                                    </div>
                                }.into_view()
                            }}
                        </div>
                    </Show>
                    
                    // Configuration Tab
                    <Show when=move || active_tab.get() == ActiveTab::Configuration>
                        <div class=move || format!("p-6 rounded-lg shadow {}",
                            if dark_mode.get() { "bg-gray-800" } else { "bg-white" }
                        )>
                            <h2 class="text-2xl font-semibold mb-6">"Simulation Configuration"</h2>
                            
                            // Scenario Templates
                            <div class="mb-8">
                                <h3 class=move || format!("text-lg font-semibold mb-4 {}",
                                    if dark_mode.get() { "text-blue-400" } else { "text-blue-600" }
                                )>"Scenario Templates"</h3>
                                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                                    {SCENARIO_TEMPLATES.iter().map(|scenario| {
                                        view! {
                                            <button
                                                on:click=move |_| apply_scenario(scenario)
                                                class=move || format!("p-4 rounded-lg border text-left transition-colors {}",
                                                    if dark_mode.get() {
                                                        "border-gray-600 hover:border-blue-500 hover:bg-gray-700"
                                                    } else {
                                                        "border-gray-200 hover:border-blue-300 hover:bg-blue-50"
                                                    }
                                                )
                                            >
                                                <h4 class="font-semibold mb-2">{scenario.name}</h4>
                                                <div class=move || format!("text-xs space-y-1 {}",
                                                    if dark_mode.get() { "text-gray-400" } else { "text-gray-600" }
                                                )>
                                                    <div>{format!("Assets: ${}", scenario.trust_assets)}</div>
                                                    <div>{format!("Redemption: {}%", scenario.redemption_percentage)}</div>
                                                    <div>{format!("Bidders: {}", scenario.num_bidders)}</div>
                                                    <div>{format!("Tokens: {}", scenario.total_tokens)}</div>
                                                </div>
                                            </button>
                                        }
                                    }).collect::<Vec<_>>()}
                                </div>
                            </div>
                            
                            // Configuration Sections
                            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                                <div>
                                    <h3 class=move || format!("text-lg font-semibold mb-4 {}",
                                        if dark_mode.get() { "text-blue-400" } else { "text-blue-600" }
                                    )>"Assets & Pool"</h3>
                                    <div class="space-y-4">
                                        <div>
                                            <label class="block text-sm font-medium mb-2">"Trust Assets"</label>
                                            <input
                                                type="number"
                                                prop:value=move || trust_assets.get()
                                                on:change=move |ev| {
                                                    if let Ok(val) = event_target_value(&ev).parse::<u32>() {
                                                        set_trust_assets.set(val);
                                                    }
                                                }
                                                class=move || format!("w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 {}",
                                                    if dark_mode.get() { "bg-gray-700 border-gray-600 text-white" } else { "bg-white border-gray-300" }
                                                )
                                            />
                                        </div>
                                        <div>
                                            <label class="block text-sm font-medium mb-2">"Redemption Percentage"</label>
                                            <input
                                                type="number"
                                                min="1"
                                                max="100"
                                                prop:value=move || redemption_percentage.get()
                                                on:change=move |ev| {
                                                    if let Ok(val) = event_target_value(&ev).parse::<u32>() {
                                                        set_redemption_percentage.set(val);
                                                    }
                                                }
                                                class=move || format!("w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 {}",
                                                    if dark_mode.get() { "bg-gray-700 border-gray-600 text-white" } else { "bg-white border-gray-300" }
                                                )
                                            />
                                            <p class=move || format!("text-sm mt-1 {}",
                                                if dark_mode.get() { "text-blue-400" } else { "text-blue-600" }
                                            )>{format!("Redemption Pool: ${}", redemption_pool())}</p>
                                        </div>
                                    </div>
                                </div>
                                
                                <div>
                                    <h3 class=move || format!("text-lg font-semibold mb-4 {}",
                                        if dark_mode.get() { "text-green-400" } else { "text-green-600" }
                                    )>"Auction Parameters"</h3>
                                    <div class="space-y-4">
                                        <div>
                                            <label class="block text-sm font-medium mb-2">"Number of Bidders"</label>
                                            <input
                                                type="number"
                                                min="2"
                                                max="50"
                                                value=move || num_bidders.get()
                                                on:change=move |ev| {
                                                    if let Ok(val) = event_target_value(&ev).parse::<usize>() {
                                                        set_num_bidders.set(val);
                                                    }
                                                }
                                                class=move || format!("w-full p-3 border rounded-lg focus:ring-2 focus:ring-green-500 {}",
                                                    if dark_mode.get() { "bg-gray-700 border-gray-600 text-white" } else { "bg-white border-gray-300" }
                                                )
                                            />
                                        </div>
                                        <div>
                                            <label class="block text-sm font-medium mb-2">"Total Tokens"</label>
                                            <input
                                                type="number"
                                                min="5"
                                                max="500"
                                                value=move || total_tokens.get()
                                                on:change=move |ev| {
                                                    if let Ok(val) = event_target_value(&ev).parse::<usize>() {
                                                        set_total_tokens.set(val);
                                                    }
                                                }
                                                class=move || format!("w-full p-3 border rounded-lg focus:ring-2 focus:ring-green-500 {}",
                                                    if dark_mode.get() { "bg-gray-700 border-gray-600 text-white" } else { "bg-white border-gray-300" }
                                                )
                                            />
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            // Growth Rates
                            <div class="mb-8">
                                <h3 class=move || format!("text-lg font-semibold mb-4 {}",
                                    if dark_mode.get() { "text-purple-400" } else { "text-purple-600" }
                                )>"Growth Rates (per auction)"</h3>
                                <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    <div>
                                        <label class="block text-sm font-medium mb-2">"Trust Assets Growth %"</label>
                                        <input
                                            type="number"
                                            min="0"
                                            max="100"
                                            value=move || trust_assets_growth.get()
                                            on:change=move |ev| {
                                                if let Ok(val) = event_target_value(&ev).parse::<u32>() {
                                                    set_trust_assets_growth.set(val);
                                                }
                                            }
                                            class=move || format!("w-full p-3 border rounded-lg focus:ring-2 focus:ring-purple-500 {}",
                                                if dark_mode.get() { "bg-gray-700 border-gray-600 text-white" } else { "bg-white border-gray-300" }
                                            )
                                        />
                                    </div>
                                    <div>
                                        <label class="block text-sm font-medium mb-2">"Redemption % Growth"</label>
                                        <input
                                            type="number"
                                            min="0"
                                            max="100"
                                            value=move || redemption_percentage_growth.get()
                                            on:change=move |ev| {
                                                if let Ok(val) = event_target_value(&ev).parse::<u32>() {
                                                    set_redemption_percentage_growth.set(val);
                                                }
                                            }
                                            class=move || format!("w-full p-3 border rounded-lg focus:ring-2 focus:ring-purple-500 {}",
                                                if dark_mode.get() { "bg-gray-700 border-gray-600 text-white" } else { "bg-white border-gray-300" }
                                            )
                                        />
                                    </div>
                                    <div>
                                        <label class="block text-sm font-medium mb-2">"Bidder Growth %"</label>
                                        <input
                                            type="number"
                                            min="0"
                                            max="100"
                                            value=move || bidder_growth.get()
                                            on:change=move |ev| {
                                                if let Ok(val) = event_target_value(&ev).parse::<u32>() {
                                                    set_bidder_growth.set(val);
                                                }
                                            }
                                            class=move || format!("w-full p-3 border rounded-lg focus:ring-2 focus:ring-purple-500 {}",
                                                if dark_mode.get() { "bg-gray-700 border-gray-600 text-white" } else { "bg-white border-gray-300" }
                                            )
                                        />
                                    </div>
                                    <div>
                                        <label class="block text-sm font-medium mb-2">"Token Growth %"</label>
                                        <input
                                            type="number"
                                            min="0"
                                            max="100"
                                            value=move || token_growth.get()
                                            on:change=move |ev| {
                                                if let Ok(val) = event_target_value(&ev).parse::<u32>() {
                                                    set_token_growth.set(val);
                                                }
                                            }
                                            class=move || format!("w-full p-3 border rounded-lg focus:ring-2 focus:ring-purple-500 {}",
                                                if dark_mode.get() { "bg-gray-700 border-gray-600 text-white" } else { "bg-white border-gray-300" }
                                            )
                                        />
                                    </div>
                                </div>
                            </div>
                            
                            // Simulation Options
                            <div class="mb-8">
                                <h3 class=move || format!("text-lg font-semibold mb-4 {}",
                                    if dark_mode.get() { "text-orange-400" } else { "text-orange-600" }
                                )>"Simulation Options"</h3>
                                <div class="space-y-4">
                                    <label class="flex items-center space-x-3">
                                        <input
                                            type="checkbox"
                                            checked=move || use_profiles.get()
                                            on:change=move |ev| {
                                                set_use_profiles.set(event_target_checked(&ev));
                                            }
                                            class="w-5 h-5 text-orange-600 border-gray-300 rounded focus:ring-orange-500"
                                        />
                                        <div>
                                            <span class="text-sm font-medium">"Use Bidder Profiles & Learning"</span>
                                            <p class=move || format!("text-xs {}",
                                                if dark_mode.get() { "text-gray-400" } else { "text-gray-500" }
                                            )>"Enable different bidding strategies that adapt over time"</p>
                                        </div>
                                    </label>
                                </div>
                            </div>
                            
                            // Apply button
                            <div class=move || format!("border-t pt-6 {}",
                                if dark_mode.get() { "border-gray-600" } else { "border-gray-200" }
                            )>
                                <button
                                    on:click=move |_| {
                                        set_active_tab.set(ActiveTab::Simulation);
                                        generate_bidders_action();
                                    }
                                    class="bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-lg font-medium transition"
                                >
                                    "Apply & Return to Simulation"
                                </button>
                            </div>
                        </div>
                    </Show>
                </div>
            </div>
        </div>
    }
}

fn main() {
    console_error_panic_hook::set_once();
    mount_to_body(|| view! { <App/> })
}