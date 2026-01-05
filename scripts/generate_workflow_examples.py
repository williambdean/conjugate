#!/usr/bin/env python3
"""
Generate images for raw-data-workflow.md examples
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure the images directory exists
images_dir = Path("docs/images")
images_dir.mkdir(exist_ok=True)


def generate_ab_testing_example():
    """Generate A/B Testing example and image"""
    from conjugate.distributions import Beta
    from conjugate.models import binomial_beta
    from conjugate.helpers import bernoulli_beta_inputs

    print("=" * 60)
    print("EXAMPLE 1: A/B Testing with Conversion Rates")
    print("=" * 60)

    # Raw observational data - user conversion outcomes
    variant_a = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]  # 8/15 conversions
    variant_b = [
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        1,
    ]  # 13/20 conversions

    # Extract sufficient statistics automatically
    inputs_a = bernoulli_beta_inputs(variant_a)
    inputs_b = bernoulli_beta_inputs(variant_b)

    print("Variant A:", inputs_a)
    print("Variant B:", inputs_b)

    # Set up priors
    prior = Beta(1, 1)

    # Compute posteriors
    posterior_a = binomial_beta(**inputs_a, prior=prior)
    posterior_b = binomial_beta(**inputs_b, prior=prior)

    print(f"Variant A: Beta({posterior_a.alpha}, {posterior_a.beta})")
    print(f"Variant B: Beta({posterior_b.alpha}, {posterior_b.beta})")

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 1, 1000)

    ax.plot(
        x, posterior_a.dist.pdf(x), label="Variant A (8/15)", linewidth=2, color="blue"
    )
    ax.plot(
        x,
        posterior_b.dist.pdf(x),
        label="Variant B (13/20)",
        linewidth=2,
        color="orange",
    )
    ax.axvline(
        inputs_a["x"] / inputs_a["n"],
        color="blue",
        alpha=0.5,
        linestyle="--",
        label="A MLE",
    )
    ax.axvline(
        inputs_b["x"] / inputs_b["n"],
        color="orange",
        alpha=0.5,
        linestyle="--",
        label="B MLE",
    )

    ax.set_xlabel("Conversion Rate")
    ax.set_ylabel("Density")
    ax.set_title("A/B Test Results: Posterior Distributions")
    ax.legend()
    ax.grid(alpha=0.3)

    # Save the plot
    plt.tight_layout()
    plt.savefig("docs/images/raw-data-ab-testing.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Statistical comparison
    prob_b_better = (posterior_b.dist.rvs(10000) > posterior_a.dist.rvs(10000)).mean()
    print(f"Probability that B > A: {prob_b_better:.3f}")
    print("✅ Example 1 complete - image saved to docs/images/raw-data-ab-testing.png")


def generate_website_analytics_example():
    """Generate Website Analytics example and image"""
    from conjugate.distributions import Gamma
    from conjugate.models import poisson_gamma, poisson_gamma_predictive
    from conjugate.helpers import poisson_gamma_inputs

    print("\n" + "=" * 60)
    print("EXAMPLE 2: Website Analytics - Daily Page Views")
    print("=" * 60)

    # Raw data: daily page views over 30 days
    daily_views = [
        1247,
        1356,
        1189,
        1445,
        1523,
        1234,
        1345,
        1456,
        1567,
        1432,
        1234,
        1345,
        1456,
        1567,
        1678,
        1789,
        1234,
        1345,
        1456,
        1567,
        1678,
        1789,
        1890,
        1234,
        1345,
        1456,
        1567,
        1678,
        1789,
        1890,
    ]

    # Extract sufficient statistics
    inputs = poisson_gamma_inputs(daily_views)
    print("Sufficient statistics:", inputs)

    # Prior: weakly informative Gamma(2, 0.001)
    prior = Gamma(alpha=2, beta=0.001)

    # Compute posterior
    posterior = poisson_gamma(**inputs, prior=prior)
    print(f"Posterior: Gamma({posterior.alpha:.1f}, {posterior.beta:.4f})")

    # Expected daily views (posterior mean)
    expected_daily_views = posterior.dist.mean()
    print(f"Expected daily views: {expected_daily_views:.0f}")

    # 95% credible interval
    lower, upper = posterior.dist.ppf([0.025, 0.975])
    print(f"95% credible interval: [{lower:.0f}, {upper:.0f}]")

    # Predictive distribution for tomorrow's views
    predictive = poisson_gamma_predictive(distribution=posterior)
    pred_mean = predictive.dist.mean()
    pred_std = predictive.dist.std()
    print(f"Tomorrow's views prediction: {pred_mean:.0f} ± {pred_std:.0f}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Daily views time series
    ax1.plot(range(1, 31), daily_views, "o-", color="steelblue", alpha=0.7)
    ax1.axhline(
        expected_daily_views,
        color="red",
        linestyle="--",
        label=f"Expected: {expected_daily_views:.0f}",
    )
    ax1.fill_between(
        range(1, 31),
        lower,
        upper,
        alpha=0.2,
        color="red",
        label=f"95% CI: [{lower:.0f}, {upper:.0f}]",
    )
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Page Views")
    ax1.set_title("Daily Page Views Over 30 Days")
    ax1.set_ylim(bottom=0)  # Include zero on y-axis
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Posterior distribution
    x = np.linspace(max(0, lower - 200), upper + 200, 1000)
    ax2.plot(x, posterior.dist.pdf(x), "r-", linewidth=2, label="Posterior")
    ax2.axvline(expected_daily_views, color="red", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Daily Views Rate (λ)")
    ax2.set_ylabel("Density")
    ax2.set_title("Posterior Distribution of Daily View Rate")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "docs/images/raw-data-website-analytics.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(
        "✅ Example 2 complete - image saved to docs/images/raw-data-website-analytics.png"
    )


def generate_survey_analysis_example():
    """Generate Customer Survey Analysis example and image"""
    from conjugate.distributions import Dirichlet
    from conjugate.models import multinomial_dirichlet
    from collections import Counter

    print("\n" + "=" * 60)
    print("EXAMPLE 3: Customer Survey Analysis")
    print("=" * 60)

    # Raw survey data: customer satisfaction ratings
    responses = [
        "Excellent",
        "Good",
        "Fair",
        "Good",
        "Excellent",
        "Fair",
        "Good",
        "Excellent",
        "Good",
        "Good",
        "Fair",
        "Poor",
        "Good",
        "Excellent",
        "Fair",
        "Good",
        "Excellent",
        "Good",
        "Fair",
        "Good",
    ]

    # Count each category
    counts = Counter(responses)
    print("Manual count verification:", dict(counts))

    # Set up categories and prior
    categories = ["Excellent", "Good", "Fair", "Poor"]
    prior_alpha = [1, 1, 1, 1]  # Uniform prior
    prior = Dirichlet(prior_alpha)

    # Convert responses to count array
    response_counts = np.array([responses.count(cat) for cat in categories])
    print("Response counts:", response_counts)

    # Compute posterior
    posterior = multinomial_dirichlet(x=response_counts, prior=prior)
    print(f"Posterior parameters: {posterior.alpha}")

    # Posterior probabilities
    posterior_probs = posterior.dist.mean()
    print("\nPosterior probabilities:")
    for cat, prob in zip(categories, posterior_probs):
        print(f"{cat}: {prob:.3f}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Response counts
    colors = ["#2E8B57", "#4682B4", "#DAA520", "#DC143C"]
    bars1 = ax1.bar(categories, response_counts, color=colors, alpha=0.7)
    ax1.set_ylabel("Count")
    ax1.set_title("Survey Response Counts")
    ax1.grid(axis="y", alpha=0.3)

    # Add count labels on bars
    for bar, count in zip(bars1, response_counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(count),
            ha="center",
            va="bottom",
        )

    # Plot 2: Posterior probabilities
    bars2 = ax2.bar(categories, posterior_probs, color=colors, alpha=0.7)
    ax2.set_ylabel("Probability")
    ax2.set_title("Posterior Probabilities")
    ax2.set_ylim(0, max(posterior_probs) * 1.2)
    ax2.grid(axis="y", alpha=0.3)

    # Add probability labels on bars
    for bar, prob in zip(bars2, posterior_probs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{prob:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(
        "docs/images/raw-data-survey-analysis.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print(
        "✅ Example 3 complete - image saved to docs/images/raw-data-survey-analysis.png"
    )


def generate_temperature_example():
    """Generate Temperature Sensor example and image"""
    from conjugate.distributions import NormalInverseGamma
    from conjugate.models import normal
    from conjugate.helpers import normal_inputs

    print("\n" + "=" * 60)
    print("EXAMPLE 4: Sensor Measurements - Temperature Data")
    print("=" * 60)

    # Raw temperature measurements from a sensor
    temperatures = [
        20.1,
        20.4,
        19.9,
        20.2,
        20.0,
        20.3,
        19.8,
        20.1,
        20.2,
        19.9,
        20.0,
        20.1,
        20.3,
        19.7,
        20.4,
        20.0,
        20.2,
        19.9,
        20.1,
        20.0,
        20.3,
        19.8,
        20.2,
        20.1,
        19.9,
        20.0,
        20.4,
        20.1,
        20.0,
        20.2,
    ]

    # Extract sufficient statistics
    inputs = normal_inputs(temperatures)
    print("Sufficient statistics:", inputs)

    # Prior: weakly informative Normal-Inverse-Gamma
    prior = NormalInverseGamma(mu=20.0, nu=1.0, alpha=2.0, beta=1.0)

    # Compute posterior
    posterior = normal(**inputs, prior=prior)

    print(f"Posterior mean (μ): {posterior.mu:.3f}")
    print(f"Posterior precision parameter (ν): {posterior.nu:.1f}")
    print(f"Posterior shape (α): {posterior.alpha:.1f}")
    print(f"Posterior rate (β): {posterior.beta:.3f}")

    # Expected temperature and variance
    expected_temp = posterior.mu
    expected_variance = (
        posterior.beta / (posterior.alpha - 1) if posterior.alpha > 1 else float("inf")
    )
    print(f"Expected temperature: {expected_temp:.3f}°C")
    print(f"Expected variance: {expected_variance:.6f}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Temperature measurements over time
    ax1.plot(
        range(1, len(temperatures) + 1),
        temperatures,
        "o-",
        color="steelblue",
        alpha=0.7,
        markersize=4,
    )
    ax1.axhline(
        np.mean(temperatures),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(temperatures):.2f}°C",
    )
    ax1.axhline(
        expected_temp,
        color="orange",
        linestyle="--",
        label=f"Expected: {expected_temp:.2f}°C",
    )
    ax1.set_xlabel("Measurement #")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Temperature Measurements Over Time")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Histogram with posterior prediction
    ax2.hist(
        temperatures,
        bins=10,
        density=True,
        alpha=0.6,
        color="lightblue",
        label="Observed Data",
    )

    # Plot posterior predictive distribution (approximate)
    x_range = np.linspace(min(temperatures) - 0.5, max(temperatures) + 0.5, 100)
    # For Normal-Inverse-Gamma, the posterior predictive is a t-distribution
    # but for simplicity, we'll approximate with a normal
    posterior_std = np.sqrt(expected_variance)
    posterior_pdf = (1 / (posterior_std * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x_range - expected_temp) / posterior_std) ** 2
    )
    ax2.plot(
        x_range, posterior_pdf, "r-", linewidth=2, label="Posterior Predictive (approx)"
    )

    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel("Density")
    ax2.set_title("Temperature Distribution")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("docs/images/raw-data-temperature.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✅ Example 4 complete - image saved to docs/images/raw-data-temperature.png")


def generate_exponential_example():
    """Generate Time Between Events example and image"""
    from conjugate.distributions import Gamma
    from conjugate.models import exponential_gamma
    from conjugate.helpers import exponential_gamma_inputs

    print("\n" + "=" * 60)
    print("EXAMPLE 5: Time Between Events - Exponential Model")
    print("=" * 60)

    # Raw data: time between customer arrivals (in minutes)
    arrival_intervals = [
        3.2,
        1.8,
        4.1,
        2.7,
        3.9,
        2.1,
        5.2,
        1.9,
        3.4,
        2.8,
        4.5,
        1.7,
        3.8,
        2.9,
        4.2,
        3.1,
        2.4,
        5.1,
        1.6,
        3.7,
        2.5,
        4.8,
        3.3,
        2.2,
        4.6,
        1.9,
        3.5,
        2.6,
        4.3,
        3.0,
    ]

    # Extract sufficient statistics
    inputs = exponential_gamma_inputs(arrival_intervals)
    print("Sufficient statistics:", inputs)

    # Prior: Gamma(2, 1) - expecting moderate arrival rate
    prior = Gamma(alpha=2.0, beta=1.0)

    # Compute posterior
    posterior = exponential_gamma(**inputs, prior=prior)

    # Posterior rate parameter (λ)
    rate = posterior.alpha / posterior.beta
    expected_interval = 1 / rate

    print(f"Posterior rate (λ): {rate:.3f} arrivals/minute")
    print(f"Expected interval: {expected_interval:.3f} minutes")

    # 95% credible interval for rate
    rate_samples = posterior.dist.rvs(10000)
    rate_ci = np.percentile(rate_samples, [2.5, 97.5])
    interval_ci = 1 / rate_ci[::-1]  # Invert and flip for interval CI

    print(f"Rate 95% CI: [{rate_ci[0]:.3f}, {rate_ci[1]:.3f}]")
    print(f"Interval 95% CI: [{interval_ci[0]:.3f}, {interval_ci[1]:.3f}] minutes")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Arrival intervals over time
    ax1.plot(
        range(1, len(arrival_intervals) + 1),
        arrival_intervals,
        "o-",
        color="steelblue",
        alpha=0.7,
        markersize=4,
    )
    ax1.axhline(
        np.mean(arrival_intervals),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(arrival_intervals):.2f} min",
    )
    ax1.axhline(
        expected_interval,
        color="orange",
        linestyle="--",
        label=f"Expected: {expected_interval:.2f} min",
    )
    ax1.set_xlabel("Customer #")
    ax1.set_ylabel("Arrival Interval (minutes)")
    ax1.set_title("Time Between Customer Arrivals")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Posterior distribution of rate parameter
    # Set range based on actual rate with some buffer
    rate_min = max(0.1, rate * 0.5)
    rate_max = rate * 1.5
    rate_range = np.linspace(rate_min, rate_max, 1000)
    ax2.plot(
        rate_range,
        posterior.dist.pdf(rate_range),
        "r-",
        linewidth=2,
        label="Posterior Rate",
    )
    ax2.axvline(
        rate, color="red", linestyle="--", alpha=0.7, label=f"Mean Rate: {rate:.3f}"
    )
    ax2.fill_between(
        rate_range, 0, posterior.dist.pdf(rate_range), alpha=0.3, color="red"
    )
    ax2.set_xlabel("Rate (λ) [arrivals/minute]")
    ax2.set_ylabel("Density")
    ax2.set_title("Posterior Distribution of Arrival Rate")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("docs/images/raw-data-exponential.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("✅ Example 5 complete - image saved to docs/images/raw-data-exponential.png")


if __name__ == "__main__":
    print("Generating images for raw-data-workflow.md examples...")
    print("This may take a few moments...\n")

    # Generate all examples
    generate_ab_testing_example()
    generate_website_analytics_example()
    generate_survey_analysis_example()
    generate_temperature_example()
    generate_exponential_example()

    print("\n" + "=" * 60)
    print("✅ ALL EXAMPLES GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print("Images saved to docs/images/:")
    print("- raw-data-ab-testing.png")
    print("- raw-data-website-analytics.png")
    print("- raw-data-survey-analysis.png")
    print("- raw-data-temperature.png")
    print("- raw-data-exponential.png")
